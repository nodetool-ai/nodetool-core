from enum import Enum
from typing import Any, List, Union


class Operator(Enum):
    """Enumeration of comparison operators supported by conditions."""

    EQ = "="
    NE = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    IN = "IN"
    LIKE = "LIKE"
    CONTAINS = "CONTAINS"


class LogicalOperator(Enum):
    """Enumeration of logical operators for combining conditions."""

    AND = "AND"
    OR = "OR"


class Variable:
    """Represents a variable placeholder in a condition's value."""

    def __init__(self, name: str):
        """
        Initializes a Variable instance.

        Args:
            name: The name of the variable.
        """
        self.name = name


class Condition:
    """Represents a single comparison condition (e.g., age > 18)."""

    def __init__(self, field: str, operator: Operator, value: Any):
        """
        Initializes a Condition instance.

        Args:
            field: The name of the field to compare.
            operator: The comparison operator to use.
            value: The value or Variable to compare against.
        """
        self.field = field
        self.operator = operator
        self.value = value


class ConditionGroup:
    """Represents a group of conditions combined by a logical operator."""

    def __init__(
        self,
        conditions: list[Union["ConditionGroup", Condition, "ConditionBuilder"]],
        operator: LogicalOperator,
    ):
        """
        Initializes a ConditionGroup instance.

        Args:
            conditions: A list of Conditions or ConditionGroups to combine.
            operator: The logical operator (AND/OR) to combine the conditions with.
        """
        self.conditions = [
            condition.build() if isinstance(condition, ConditionBuilder) else condition for condition in conditions
        ]
        self.operator = operator


class Field:
    """A fluent interface for creating conditions related to a specific field."""

    def __init__(self, name: str):
        """
        Initializes a Field instance.

        Args:
            name: The name of the field this instance represents.
        """
        self.name = name

    def _create_condition(self, operator: Operator, value: Any | Variable) -> "ConditionBuilder":
        """
        Internal helper method to create a ConditionBuilder from a condition.

        Args:
            operator: The comparison operator.
            value: The value or Variable for the comparison.

        Returns:
            A ConditionBuilder instance representing the created condition.
        """
        return ConditionBuilder(Condition(self.name, operator, value))

    def equals(self, value: Any | Variable) -> "ConditionBuilder":
        """Creates an 'equals' (==) condition."""
        return self._create_condition(Operator.EQ, value)

    def not_equals(self, value: Any | Variable) -> "ConditionBuilder":
        """Creates a 'not equals' (!=) condition."""
        return self._create_condition(Operator.NE, value)

    def greater_than(self, value: Any | Variable) -> "ConditionBuilder":
        """Creates a 'greater than' (>) condition."""
        return self._create_condition(Operator.GT, value)

    def less_than(self, value: Any | Variable) -> "ConditionBuilder":
        """Creates a 'less than' (<) condition."""
        return self._create_condition(Operator.LT, value)

    def greater_than_or_equal(self, value: Any | Variable) -> "ConditionBuilder":
        """Creates a 'greater than or equal to' (>=) condition."""
        return self._create_condition(Operator.GTE, value)

    def less_than_or_equal(self, value: Any | Variable) -> "ConditionBuilder":
        """Creates a 'less than or equal to' (<=) condition."""
        return self._create_condition(Operator.LTE, value)

    def in_list(self, values: list[Any] | Variable) -> "ConditionBuilder":
        """Creates an 'in list' (IN) condition."""
        return self._create_condition(Operator.IN, values)

    def like(self, pattern: str | Variable) -> "ConditionBuilder":
        """Creates a 'like' (LIKE) condition for pattern matching."""
        return self._create_condition(Operator.LIKE, pattern)

    # --- Operator Overloads ---

    def __eq__(self, value: Any | Variable) -> "ConditionBuilder":
        """Overloads the '==' operator to create an 'equals' condition."""
        return self.equals(value)

    def __ne__(self, value: Any | Variable) -> "ConditionBuilder":
        """Overloads the '!=' operator to create a 'not equals' condition."""
        return self.not_equals(value)

    def __gt__(self, value: Any | Variable) -> "ConditionBuilder":
        """Overloads the '>' operator to create a 'greater than' condition."""
        return self.greater_than(value)

    def __lt__(self, value: Any | Variable) -> "ConditionBuilder":
        """Overloads the '<' operator to create a 'less than' condition."""
        return self.less_than(value)

    def __ge__(self, value: Any | Variable) -> "ConditionBuilder":
        """Overloads the '>=' operator to create a 'greater than or equal to' condition."""
        return self.greater_than_or_equal(value)

    def __le__(self, value: Any | Variable) -> "ConditionBuilder":
        """Overloads the '<=' operator to create a 'less than or equal to' condition."""
        return self.less_than_or_equal(value)

    def __contains__(self, value: list[Any] | Variable) -> "ConditionBuilder":
        """
        Overloads the 'in' operator (used as `value in field`)
        to create an 'in list' condition.
        Note: The typical Python `in` usage is reversed here for fluent API:
        `Field("tags").__contains__(["urgent", "important"])` is equivalent to
        `Field("tags").in_list(["urgent", "important"])`.
        Direct use `value in Field("tags")` is not the intended pattern.
        """
        # The Operator.CONTAINS is often associated with checking if a field
        # contains a substring or element. Using Operator.IN here aligns
        # with the `in_list` method's functionality. If true 'contains'
        # semantics are needed, a separate method/operator might be required.
        return self.in_list(value)


class ConditionBuilder:
    """
    Provides a fluent API for building complex condition trees using AND/OR logic.

    Starts with a single Condition created via a Field object and allows chaining
    additional conditions using `and_` or `or_`.
    """

    def __init__(self, condition: Condition | ConditionGroup):
        """
        Initializes the ConditionBuilder with the first condition.

        Args:
            condition: The initial Condition or ConditionGroup to start building from.
                       Typically created using a Field object (e.g., Field("age") > 18).
        """
        if isinstance(condition, Condition):
            self.root = ConditionGroup([condition], LogicalOperator.AND)
        else:
            self.root = condition

    def _add_condition(self, other: "ConditionBuilder", operator: LogicalOperator) -> "ConditionBuilder":
        """
        Internal helper to add another condition or group using a logical operator.

        Handles combining the current root structure with the 'other' ConditionBuilder's
        root structure based on the specified logical operator. It optimizes
        for the simple case of combining two single conditions.

        Args:
            other: The ConditionBuilder representing the condition(s) to add.
            operator: The logical operator (AND/OR) to combine with.

        Returns:
            The current ConditionBuilder instance for method chaining.
        """
        # Check if the current root is just a single condition
        if (
            isinstance(self.root, ConditionGroup)
            and len(self.root.conditions) == 1
            and isinstance(self.root.conditions[0], Condition)
            and self.root.operator == LogicalOperator.AND  # Initial state
        ):
            # Simple case: combine two single conditions or a condition and a group
            # Ensure 'other.root' is also unwrapped if it's a simple single-condition group
            other_condition = other.root
            if (
                isinstance(other.root, ConditionGroup)
                and len(other.root.conditions) == 1
                and other.root.operator == LogicalOperator.AND
            ):  # Initial state of other
                other_condition = other.root.conditions[0]

            # If the new operator matches the existing group's operator (or if it's the first combination),
            # just add to the list. Otherwise, create a new group.
            # This logic needs refinement to handle nested AND/OR correctly.

            # Simplified logic for now: always create a group if combining
            # More complex logic would involve checking operator precedence and associativity
            # For now, treat the initial state as ready to form a group
            self.root = ConditionGroup([self.root.conditions[0], other_condition], operator)

        # Check if the 'other' root is just a single condition
        elif (
            isinstance(other.root, ConditionGroup)
            and len(other.root.conditions) == 1
            and isinstance(other.root.conditions[0], Condition)
            and other.root.operator == LogicalOperator.AND  # Initial state of other
        ):
            # Combine the existing group/condition with the single condition from 'other'
            self.root = ConditionGroup([self.root, other.root.conditions[0]], operator)

        else:
            # General case: combine two groups (or a group and a condition treated as a group)
            new_root = ConditionGroup([self.root, other.root], operator)
            self.root = new_root
        return self

    def and_(self, other: "ConditionBuilder") -> "ConditionBuilder":
        """
        Combines the current condition structure with another using logical AND.

        Args:
            other: The ConditionBuilder representing the condition(s) to AND with.

        Returns:
            The current ConditionBuilder instance for method chaining.
        """
        return self._add_condition(other, LogicalOperator.AND)

    def or_(self, other: "ConditionBuilder") -> "ConditionBuilder":
        """
        Combines the current condition structure with another using logical OR.

        Args:
            other: The ConditionBuilder representing the condition(s) to OR with.

        Returns:
            The current ConditionBuilder instance for method chaining.
        """
        return self._add_condition(other, LogicalOperator.OR)

    def build(self) -> ConditionGroup:
        """
        Finalizes the building process and returns the resulting ConditionGroup tree.

        Returns:
            The root ConditionGroup representing the constructed condition logic.
        """
        return self.root

    def _get_variables(self, values: dict[str, Any], condition: Condition | ConditionGroup):
        """
        Recursively traverses the condition tree to find all Variable instances.

        Args:
            values: A dictionary to populate with variable names found.
            condition: The current Condition or ConditionGroup being processed.
        """
        if isinstance(condition, Condition):
            if isinstance(condition.value, Variable):
                # Store variable names, typically initializing value to None
                values[condition.value.name] = None
            # Also check if the 'value' for IN or CONTAINS is a Variable list
            elif isinstance(condition.value, list) and any(isinstance(v, Variable) for v in condition.value):
                for item in condition.value:
                    if isinstance(item, Variable):
                        values[item.name] = None
            # Handle case where IN operator uses a single Variable representing a list
            elif condition.operator == Operator.IN and isinstance(condition.value, Variable):
                values[condition.value.name] = None

        elif isinstance(condition, ConditionGroup):  # Check if it's a ConditionGroup
            for sub_condition in condition.conditions:
                self._get_variables(values, sub_condition)
        # else: condition is likely a primitive type or non-variable object, ignore

    def get_variables(self) -> dict[str, Any]:
        """
        Extracts all unique variable names used within the condition structure.

        Returns:
            A dictionary where keys are the names of the variables found.
            The values are initialized to None, intended to be filled later.
        """
        values: dict[str, Any] = {}
        self._get_variables(values, self.root)
        return values


# Example usage:
# condition = (
#     Field("age").greater_than(Variable("min_age"))
#     .and_(Field("status").equals("active"))
#     .or_(Field("priority").in_list([Variable("high_priority"), "critical"]))
# )
# built_condition = condition.build()
# variables_needed = condition.get_variables() # -> {"min_age": None, "high_priority": None}
# print(variables_needed)
# # You would then populate these variables before evaluating the condition
# evaluation_context = {"min_age": 18, "high_priority": 5}
