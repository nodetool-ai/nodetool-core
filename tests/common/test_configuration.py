from nodetool.common.configuration import register_setting


def test_register_setting():
    regs = register_setting(
        package_name="testpkg",
        env_var="VAR1",
        group="general",
        description="desc",
        is_secret=False,
    )
    assert any(s.env_var == "VAR1" and s.package_name == "testpkg" for s in regs)
