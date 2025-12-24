"""
Math Tools for Nodetool Agents

This module provides mathematical calculation tools that allow agents to perform
various mathematical operations, from basic arithmetic to advanced calculations.

Tools included:
- CalculatorTool: Basic arithmetic operations
- StatisticsTool: Statistical calculations (mean, median, std dev, etc.)
- GeometryTool: Geometric calculations (area, volume, distance)
- AlgebraTool: Algebraic operations (solving equations, polynomials)
- TrigonometryTool: Trigonometric functions
- ConversionTool: Unit conversions
"""

import math
import statistics
from typing import Any, ClassVar, Dict

from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext


class CalculatorTool(Tool):
    """Basic arithmetic calculator tool for mathematical expressions."""

    name: str = "calculator"
    description: str = (
        "Performs basic arithmetic calculations. "
        "Supports +, -, *, /, **, sqrt, abs, round, and parentheses. "
        "Use for evaluating mathematical expressions safely."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'abs(-5)')",
            }
        },
        "required": ["expression"],
    }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        expression = params["expression"].strip()

        try:
            # Replace common math functions
            safe_expression = expression.lower()
            safe_expression = safe_expression.replace("sqrt", "math.sqrt")
            safe_expression = safe_expression.replace("abs", "abs")
            safe_expression = safe_expression.replace("round", "round")
            safe_expression = safe_expression.replace("^", "**")  # Convert ^ to **

            # Only allow safe characters and functions
            allowed_chars = set("0123456789+-*/.() mathsqrtabsround,")
            if not all(c in allowed_chars for c in safe_expression.replace(" ", "")):
                return {"error": "Expression contains invalid characters"}

            # Evaluate the expression
            result = eval(safe_expression, {"__builtins__": {}, "math": math}, {})

            return {
                "expression": expression,
                "result": result,
                "formatted_result": f"{expression} = {result}",
            }

        except Exception as e:
            return {"expression": expression, "error": f"Calculation error: {str(e)}"}


class StatisticsTool(Tool):
    """Statistical calculations tool."""

    name: str = "statistics"
    description: str = (
        "Performs statistical calculations on numerical data. "
        "Calculates mean, median, mode, standard deviation, variance, min, max, sum, and count."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Array of numerical values to analyze",
            },
            "calculations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "mean",
                        "median",
                        "mode",
                        "std_dev",
                        "variance",
                        "min",
                        "max",
                        "sum",
                        "count",
                        "all",
                    ],
                },
                "description": "Which statistics to calculate (use 'all' for all statistics)",
                "default": ["all"],
            },
        },
        "required": ["data"],
    }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        data = params["data"]
        calculations = params.get("calculations", ["all"])

        if not data:
            return {"error": "No data provided"}

        try:
            results = {}

            if "all" in calculations or "mean" in calculations:
                results["mean"] = statistics.mean(data)

            if "all" in calculations or "median" in calculations:
                results["median"] = statistics.median(data)

            if "all" in calculations or "mode" in calculations:
                try:
                    results["mode"] = statistics.mode(data)
                except statistics.StatisticsError:
                    results["mode"] = "No unique mode"

            if "all" in calculations or "std_dev" in calculations:
                if len(data) > 1:
                    results["std_dev"] = statistics.stdev(data)
                else:
                    results["std_dev"] = 0

            if "all" in calculations or "variance" in calculations:
                if len(data) > 1:
                    results["variance"] = statistics.variance(data)
                else:
                    results["variance"] = 0

            if "all" in calculations or "min" in calculations:
                results["min"] = min(data)

            if "all" in calculations or "max" in calculations:
                results["max"] = max(data)

            if "all" in calculations or "sum" in calculations:
                results["sum"] = sum(data)

            if "all" in calculations or "count" in calculations:
                results["count"] = len(data)

            return {
                "data_summary": f"Analyzed {len(data)} values",
                "statistics": results,
            }

        except Exception as e:
            return {"error": f"Statistics calculation error: {str(e)}"}


class GeometryTool(Tool):
    """Geometric calculations tool."""

    name: str = "geometry"
    description: str = (
        "Performs geometric calculations for various shapes. "
        "Supports area and perimeter for 2D shapes, volume and surface area for 3D shapes, "
        "and distance calculations."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "shape": {
                "type": "string",
                "enum": [
                    "circle",
                    "rectangle",
                    "triangle",
                    "sphere",
                    "cylinder",
                    "cube",
                    "distance_2d",
                    "distance_3d",
                ],
                "description": "Type of geometric shape or calculation",
            },
            "dimensions": {
                "type": "object",
                "description": "Dimensions specific to the shape",
                "properties": {
                    "radius": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "length": {"type": "number"},
                    "base": {"type": "number"},
                    "side_a": {"type": "number"},
                    "side_b": {"type": "number"},
                    "side_c": {"type": "number"},
                    "x1": {"type": "number"},
                    "y1": {"type": "number"},
                    "z1": {"type": "number"},
                    "x2": {"type": "number"},
                    "y2": {"type": "number"},
                    "z2": {"type": "number"},
                },
            },
        },
        "required": ["shape", "dimensions"],
    }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        shape = params["shape"]
        dims = params["dimensions"]

        try:
            results = {"shape": shape}

            if shape == "circle":
                radius = dims["radius"]
                results["area"] = math.pi * radius**2
                results["circumference"] = 2 * math.pi * radius
                results["diameter"] = 2 * radius

            elif shape == "rectangle":
                width = dims["width"]
                height = dims["height"]
                results["area"] = width * height
                results["perimeter"] = 2 * (width + height)
                results["diagonal"] = math.sqrt(width**2 + height**2)

            elif shape == "triangle":
                if "base" in dims and "height" in dims:
                    results["area"] = 0.5 * dims["base"] * dims["height"]
                if all(side in dims for side in ["side_a", "side_b", "side_c"]):
                    a, b, c = dims["side_a"], dims["side_b"], dims["side_c"]
                    results["perimeter"] = a + b + c
                    # Heron's formula
                    s = (a + b + c) / 2
                    results["area_herons"] = math.sqrt(s * (s - a) * (s - b) * (s - c))

            elif shape == "sphere":
                radius = dims["radius"]
                results["volume"] = (4 / 3) * math.pi * radius**3
                results["surface_area"] = 4 * math.pi * radius**2

            elif shape == "cylinder":
                radius = dims["radius"]
                height = dims["height"]
                results["volume"] = math.pi * radius**2 * height
                results["surface_area"] = 2 * math.pi * radius * (radius + height)

            elif shape == "cube":
                side = dims.get("side", dims.get("width", dims.get("length")))
                results["volume"] = side**3
                results["surface_area"] = 6 * side**2

            elif shape == "distance_2d":
                x1, y1 = dims["x1"], dims["y1"]
                x2, y2 = dims["x2"], dims["y2"]
                results["distance"] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            elif shape == "distance_3d":
                x1, y1, z1 = dims["x1"], dims["y1"], dims["z1"]
                x2, y2, z2 = dims["x2"], dims["y2"], dims["z2"]
                results["distance"] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

            return results

        except Exception as e:
            return {"error": f"Geometry calculation error: {str(e)}"}


class TrigonometryTool(Tool):
    """Trigonometric functions tool."""

    name: str = "trigonometry"
    description: str = (
        "Performs trigonometric calculations. "
        "Supports sin, cos, tan, asin, acos, atan, and angle conversions between degrees and radians."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "function": {
                "type": "string",
                "enum": [
                    "sin",
                    "cos",
                    "tan",
                    "asin",
                    "acos",
                    "atan",
                    "deg_to_rad",
                    "rad_to_deg",
                ],
                "description": "Trigonometric function to apply",
            },
            "value": {
                "type": "number",
                "description": "Input value (angle in degrees for trig functions, or value for inverse functions)",
            },
            "angle_unit": {
                "type": "string",
                "enum": ["degrees", "radians"],
                "description": "Unit of the input angle",
                "default": "degrees",
            },
        },
        "required": ["function", "value"],
    }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        function = params["function"]
        value = params["value"]
        angle_unit = params.get("angle_unit", "degrees")

        try:
            # Convert to radians if needed
            if function in ["sin", "cos", "tan"] and angle_unit == "degrees":
                input_radians = math.radians(value)
            else:
                input_radians = value

            result = None
            if function == "sin":
                result = math.sin(input_radians)
            elif function == "cos":
                result = math.cos(input_radians)
            elif function == "tan":
                result = math.tan(input_radians)
            elif function == "asin":
                result = math.asin(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
            elif function == "acos":
                result = math.acos(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
            elif function == "atan":
                result = math.atan(value)
                if angle_unit == "degrees":
                    result = math.degrees(result)
            elif function == "deg_to_rad":
                result = math.radians(value)
            elif function == "rad_to_deg":
                result = math.degrees(value)

            if result is None:
                raise ValueError(f"Unsupported function: {function}")

            return {
                "function": function,
                "input_value": value,
                "input_unit": angle_unit,
                "result": result,
            }

        except Exception as e:
            return {"error": f"Trigonometry calculation error: {str(e)}"}


class ConversionTool(Tool):
    """Unit conversion tool."""

    name: str = "unit_conversion"
    description: str = (
        "Converts between different units of measurement. "
        "Supports length, weight, temperature, area, and volume conversions."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "Value to convert"},
            "from_unit": {
                "type": "string",
                "description": "Source unit (e.g., 'meters', 'feet', 'celsius', 'kg')",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit (e.g., 'feet', 'meters', 'fahrenheit', 'lbs')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    def __init__(self):
        super().__init__()
        # Conversion factors to base units
        self.conversions = {
            # Length (to meters)
            "meters": 1.0,
            "m": 1.0,
            "centimeters": 0.01,
            "cm": 0.01,
            "millimeters": 0.001,
            "mm": 0.001,
            "kilometers": 1000.0,
            "km": 1000.0,
            "feet": 0.3048,
            "ft": 0.3048,
            "inches": 0.0254,
            "in": 0.0254,
            "yards": 0.9144,
            "yd": 0.9144,
            "miles": 1609.344,
            # Weight (to kilograms)
            "kilograms": 1.0,
            "kg": 1.0,
            "grams": 0.001,
            "g": 0.001,
            "pounds": 0.453592,
            "lbs": 0.453592,
            "ounces": 0.0283495,
            "oz": 0.0283495,
            # Area (to square meters)
            "square_meters": 1.0,
            "m2": 1.0,
            "square_feet": 0.092903,
            "ft2": 0.092903,
            "square_inches": 0.00064516,
            "in2": 0.00064516,
            "acres": 4046.86,
            # Volume (to liters)
            "liters": 1.0,
            "l": 1.0,
            "milliliters": 0.001,
            "ml": 0.001,
            "gallons": 3.78541,
            "gal": 3.78541,
            "cups": 0.236588,
            "fluid_ounces": 0.0295735,
            "fl_oz": 0.0295735,
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Dict[str, Any]:
        value = params["value"]
        from_unit = params["from_unit"].lower()
        to_unit = params["to_unit"].lower()

        try:
            # Special case for temperature
            if from_unit in ["celsius", "fahrenheit", "kelvin"] or to_unit in [
                "celsius",
                "fahrenheit",
                "kelvin",
            ]:
                result = self._convert_temperature(value, from_unit, to_unit)
            else:
                # Regular unit conversion
                if from_unit not in self.conversions or to_unit not in self.conversions:
                    return {"error": f"Unsupported unit conversion: {from_unit} to {to_unit}"}

                # Convert to base unit, then to target unit
                base_value = value * self.conversions[from_unit]
                result = base_value / self.conversions[to_unit]

            return {
                "original_value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "converted_value": result,
                "formatted": f"{value} {from_unit} = {result} {to_unit}",
            }

        except Exception as e:
            return {"error": f"Conversion error: {str(e)}"}

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between celsius, fahrenheit, and kelvin."""
        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "kelvin":
            celsius = value - 273.15
        else:  # celsius
            celsius = value

        # Convert from Celsius to target
        if to_unit == "fahrenheit":
            return celsius * 9 / 5 + 32
        elif to_unit == "kelvin":
            return celsius + 273.15
        else:  # celsius
            return celsius
