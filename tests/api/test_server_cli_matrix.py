import os

from click.testing import CliRunner

from nodetool.cli import cli


class TestServeCliMatrix:
    def test_non_production_forwards_mode_auth_and_feature_flags(self, monkeypatch):
        create_app_calls = []
        uvicorn_calls = []

        def mock_create_app(*args, **kwargs):
            create_app_calls.append({"args": args, "kwargs": kwargs})
            from fastapi import FastAPI

            return FastAPI()

        def mock_run_uvicorn_server(*, app, host, port, reload):
            uvicorn_calls.append({"host": host, "port": port, "reload": reload, "app": app})

        monkeypatch.setattr("nodetool.api.server.create_app", mock_create_app)
        monkeypatch.setattr("nodetool.api.server.run_uvicorn_server", mock_run_uvicorn_server)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "serve",
                "--mode",
                "public",
                "--auth-provider",
                "supabase",
                "--disable-terminal-ws",
                "--disable-deploy-admin",
                "--disable-hf-download-ws",
                "--port",
                "9901",
            ],
        )

        assert result.exit_code == 0
        assert len(create_app_calls) == 1
        kwargs = create_app_calls[0]["kwargs"]
        assert kwargs["mode"] == "public"
        assert kwargs["auth_provider"] == "supabase"
        assert kwargs["enable_terminal_ws"] is False
        assert kwargs["include_deploy_admin_router"] is False
        assert kwargs["enable_hf_download_ws"] is False
        assert len(uvicorn_calls) == 1
        assert uvicorn_calls[0]["port"] == 9901

    def test_production_defaults_to_private_mode_via_env(self, monkeypatch):
        run_server_calls = []

        def mock_run_server(**kwargs):
            run_server_calls.append({"kwargs": kwargs, "mode_env": os.environ.get("NODETOOL_SERVER_MODE")})

        monkeypatch.setattr("nodetool.api.run_server.run_server", mock_run_server)

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--production", "--port", "9902"])

        assert result.exit_code == 0
        assert len(run_server_calls) == 1
        assert run_server_calls[0]["kwargs"]["port"] == 9902
        assert run_server_calls[0]["mode_env"] == "private"

    def test_production_explicit_mode_is_forwarded(self, monkeypatch):
        run_server_calls = []

        def mock_run_server(**kwargs):
            run_server_calls.append(kwargs)

        monkeypatch.setattr("nodetool.api.run_server.run_server", mock_run_server)

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--production", "--mode", "public", "--port", "9903"])

        assert result.exit_code == 0
        assert len(run_server_calls) == 1
        assert run_server_calls[0]["mode"] == "public"
        assert run_server_calls[0]["port"] == 9903
