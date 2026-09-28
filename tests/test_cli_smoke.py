from __future__ import annotations

from typer.testing import CliRunner

from nexusagent.cli.app import app


def test_cli_shows_help_without_task() -> None:
    runner = CliRunner()

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "nexusagent" in result.output


def test_cli_accepts_max_attempts_option_without_task() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--max-attempts", "2"])

    assert result.exit_code == 0
    assert "nexusagent" in result.output


def test_cli_accepts_approval_mode_option_without_task() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--approval-mode", "deny"])

    assert result.exit_code == 0
    assert "nexusagent" in result.output


def test_cli_accepts_checkpoint_mode_option_without_task() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--checkpoint-mode", "strict"])

    assert result.exit_code == 0
    assert "nexusagent" in result.output


def test_cli_accepts_trace_mode_option_without_task() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--trace-mode", "off"])

    assert result.exit_code == 0
    assert "nexusagent" in result.output


def test_cli_accepts_resume_option_without_task(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    calls = []

    def fake_stream(*args, **kwargs):
        calls.append((args, kwargs))
        yield {"type": "workspace", "path": str(tmp_path)}

    monkeypatch.setattr("nexusagent.cli.app.stream_agent_events", fake_stream)
    result = runner.invoke(app, ["--resume", str(tmp_path)])

    assert result.exit_code == 0
    assert calls
    assert calls[0][1]["resume_workspace"] == tmp_path


def test_cli_passes_trace_mode(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    calls = []

    def fake_stream(*args, **kwargs):
        calls.append((args, kwargs))
        yield {"type": "workspace", "path": str(tmp_path)}

    monkeypatch.setattr("nexusagent.cli.app.stream_agent_events", fake_stream)
    result = runner.invoke(app, ["--trace-mode", "off", "demo task"])

    assert result.exit_code == 0
    assert calls
    assert calls[0][1]["trace_mode"] == "off"
