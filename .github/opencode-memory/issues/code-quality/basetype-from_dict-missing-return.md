# BaseType.from_dict Missing Return Statement

**Problem**: The `BaseType.from_dict()` method in `src/nodetool/metadata/types.py` was missing its return statement, causing it to return `None` instead of creating an instance.

**Solution**: Add `return NameToType[type_name](**data)` at the end of the method.

**Why**: This broke property assignment in `BaseNode.assign_property()` when trying to convert dicts with `type` field to BaseType instances like `DataframeRef` and `ImageRef`. The method was validating the type but not returning the created instance.

**Files**:
- `src/nodetool/metadata/types.py:89-105` (BaseType.from_dict method)

**Related Tests**:
- `tests/workflows/test_base_node.py::test_node_assign_property_with_dataframe_dict`
- `tests/workflows/test_base_node.py::test_node_set_properties_with_complex_types`
- `tests/workflows/test_base_node.py::test_node_assign_property_uses_from_dict_for_base_types`
- `tests/workflows/test_base_node.py::test_node_from_dict_with_base_type_properties`
- `tests/workflows/test_base_node.py::test_node_assign_property_list_of_base_types`

**Date**: 2026-02-08
