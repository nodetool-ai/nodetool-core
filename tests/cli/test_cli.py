from click.testing import CliRunner

from nodetool.cli import cli
from nodetool.config.settings import load_settings


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Nodetool CLI" in result.output


def test_show_settings(monkeypatch):
    def fake_load_settings():
        return {"FONT_PATH": "/fonts"}, {}

    monkeypatch.setattr("nodetool.config.settings.load_settings", fake_load_settings)
    runner = CliRunner()
    result = runner.invoke(cli, ["settings", "show"])
    assert result.exit_code == 0
    assert "Settings" in result.output
    assert "/fonts" in result.output


def test_show_secrets_mask(monkeypatch):
    def fake_load_settings():
        return {}, {"OPENAI_API_KEY": "secret"}

    monkeypatch.setattr("nodetool.config.settings.load_settings", fake_load_settings)
    runner = CliRunner()
    result = runner.invoke(cli, ["settings", "show", "--secrets", "--mask"])
    assert result.exit_code == 0
    assert "Secrets" in result.output
    assert "****" in result.output


def test_package_list(monkeypatch):
    from nodetool.metadata.node_metadata import PackageModel

    dummy_package = PackageModel(
        name="demo",
        description="desc",
        version="0.1",
        authors=[],
        namespaces=[],
        repo_id="owner/demo",
    )

    class DummyRegistry:
        def list_installed_packages(self):
            return [dummy_package]

    monkeypatch.setattr("nodetool.packages.registry.Registry", DummyRegistry)
    runner = CliRunner()
    result = runner.invoke(cli, ["package", "list"])
    assert result.exit_code == 0
    assert "Installed Packages" in result.output
    assert "demo" in result.output
