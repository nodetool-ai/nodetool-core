from nodetool.metadata.type_metadata import TypeMetadata

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from typing import Any, Optional


class Property(BaseModel):
    """
    Property of a node.

    This class represents a property of a node with type information, constraints,
    and metadata. It can be used to generate JSON schema and can be created from
    a Pydantic field.

    Attributes:
        name: The name of the property
        type: Type metadata for the property
        default: Default value for the property, if any
        title: Human-readable title for the property
        description: Detailed description of the property
        min: Minimum allowed value for numeric properties
        max: Maximum allowed value for numeric properties
    """

    name: str
    type: TypeMetadata
    default: Optional[Any] = None
    title: Optional[str] = None
    description: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def get_json_schema(self):
        """
        Returns a JSON schema for the property.

        Generates a JSON schema representation of this property based on its type
        and constraints. Includes description, minimum, and maximum values if set.

        Returns:
            dict: A JSON schema dictionary representing this property
        """
        schema = self.type.get_json_schema()
        if self.description and self.description != "":
            schema["description"] = self.description
        if self.min:
            schema["minimum"] = self.min if self.type.type != "int" else int(self.min)
        if self.max:
            schema["maximum"] = self.max if self.type.type != "int" else int(self.max)
        return schema

    @staticmethod
    def from_field(name: str, type_: TypeMetadata, field: FieldInfo):
        """
        Creates a Property instance from a Pydantic field.

        Extracts metadata, constraints, and other information from a Pydantic
        FieldInfo object to create a Property instance.

        Args:
            name: The name of the property
            type_: TypeMetadata object representing the property's type
            field: Pydantic FieldInfo object containing field metadata

        Returns:
            Property: A new Property instance with data extracted from the field
        """
        metadata = {type(f): f for f in field.metadata}

        ge = metadata.get(annotated_types.Ge, None)
        le = metadata.get(annotated_types.Le, None)

        if field.title is None:
            title = name.replace("_", " ").title()
        else:
            title = field.title
        return Property(
            name=name,
            type=type_,
            default=field.default,
            title=title,
            description=field.description,
            min=ge.ge if ge is not None else None,
            max=le.le if le is not None else None,
        )
