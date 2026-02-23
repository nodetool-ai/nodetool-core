from nodetool.types.wrap_primitive_types import wrap_primitive_types

def test_wrap_string():
    assert wrap_primitive_types("hello") == {"type": "string", "value": "hello"}
    assert wrap_primitive_types("") == {"type": "string", "value": ""}

def test_wrap_integer():
    assert wrap_primitive_types(123) == {"type": "integer", "value": 123}
    assert wrap_primitive_types(0) == {"type": "integer", "value": 0}
    assert wrap_primitive_types(-1) == {"type": "integer", "value": -1}

def test_wrap_float():
    assert wrap_primitive_types(1.23) == {"type": "float", "value": 1.23}
    assert wrap_primitive_types(0.0) == {"type": "float", "value": 0.0}
    assert wrap_primitive_types(-1.5) == {"type": "float", "value": -1.5}

def test_wrap_boolean():
    assert wrap_primitive_types(True) == {"type": "boolean", "value": True}
    assert wrap_primitive_types(False) == {"type": "boolean", "value": False}

def test_wrap_bytes():
    assert wrap_primitive_types(b"hello") == {"type": "bytes", "value": b"hello"}
    assert wrap_primitive_types(b"") == {"type": "bytes", "value": b""}

def test_wrap_list():
    assert wrap_primitive_types([1, "two"]) == {
        "type": "list",
        "value": [{"type": "integer", "value": 1}, {"type": "string", "value": "two"}],
    }
    assert wrap_primitive_types([]) == {"type": "list", "value": []}

def test_wrap_dict():
    # Dict without "type" key
    assert wrap_primitive_types({"a": 1, "b": "c"}) == {
        "a": {"type": "integer", "value": 1},
        "b": {"type": "string", "value": "c"},
    }
    assert wrap_primitive_types({}) == {}

    # Dict with "type" key
    already_wrapped = {"type": "custom", "value": "val"}
    assert wrap_primitive_types(already_wrapped) == already_wrapped

def test_wrap_nested():
    nested = {"list": [1, {"a": 2}], "b": 3}
    expected = {
        "list": {
            "type": "list",
            "value": [
                {"type": "integer", "value": 1},
                {"a": {"type": "integer", "value": 2}},
            ],
        },
        "b": {"type": "integer", "value": 3},
    }
    assert wrap_primitive_types(nested) == expected

def test_wrap_other_types():
    assert wrap_primitive_types(None) is None

    class Custom:
        pass

    custom_obj = Custom()
    assert wrap_primitive_types(custom_obj) == custom_obj
