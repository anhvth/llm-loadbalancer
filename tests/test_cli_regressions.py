import subprocess

from llm_loadbalancer import cli as main, keep_connection


def test_default_config_path_uses_cache_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    assert main.default_config_path() == tmp_path / ".config" / "llm-proxy.yaml"


def test_ensure_config_exists_creates_expected_default_file(tmp_path):
    config_path = tmp_path / "config.yaml"

    main.ensure_config_exists(config_path)

    assert config_path.read_text() == main.DEFAULT_CONFIG


def test_ensure_config_exists_creates_parent_directories(tmp_path):
    config_path = tmp_path / ".config" / "llm-proxy.yaml"

    main.ensure_config_exists(config_path)

    assert config_path.exists()


def test_ensure_config_exists_does_not_overwrite_existing_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("custom-config\n")

    main.ensure_config_exists(config_path)

    assert config_path.read_text() == "custom-config\n"


def test_open_config_in_editor_creates_config_and_uses_editor(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    calls = []

    def fake_call(argv):
        calls.append(argv)
        return 0

    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.setattr(main.subprocess, "call", fake_call)

    result = main.open_config_in_editor(config_path)

    assert result == 0
    assert config_path.read_text() == main.DEFAULT_CONFIG
    assert calls == [["true", str(config_path)]]


def test_main_set_config_uses_explicit_config_path(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    seen = []

    def fake_open_config_in_editor(path):
        seen.append(path)
        return 0

    monkeypatch.setattr(main, "open_config_in_editor", fake_open_config_in_editor)

    result = main.main(["--set-config", "--config", str(config_path)])

    assert result == 0
    assert seen == [config_path]


def test_main_defaults_to_cache_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = []

    def fake_start_everything(path, verbose=False):
        seen.append((path, verbose))
        return 0

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main([])

    assert result == 0
    assert seen == [(tmp_path / ".config" / "llm-proxy.yaml", True)]


def test_main_passes_silent_flag_to_start_everything(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    seen = []

    def fake_start_everything(path, verbose=False):
        seen.append((path, verbose))
        return 0

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main(["--config", str(config_path), "--silent"])

    assert result == 0
    assert seen == [(config_path, False)]


def test_main_passes_routing_override_to_start_everything(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    seen = []

    def fake_start_everything(path, verbose=False, routing=None):
        seen.append((path, verbose, routing))
        return 0

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main(["--config", str(config_path), "--routing", "random"])

    assert result == 0
    assert seen == [(config_path, True, "random")]


def test_start_everything_launches_tmux_before_load_balancer(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(main.DEFAULT_CONFIG)
    call_order = []

    def fake_launch_in_tmux(session_name, commands):
        call_order.append(("tmux", session_name, commands))

    def fake_serve_forever(path, verbose=False):
        call_order.append(("serve", path, verbose))

    monkeypatch.setattr(main, "launch_in_tmux", fake_launch_in_tmux)
    monkeypatch.setattr(main, "serve_forever", fake_serve_forever)

    result = main.start_everything(config_path)

    assert result == 0
    assert call_order[0][0] == "tmux"
    assert call_order[1] == ("serve", config_path, False)
    assert call_order[0][1] == "keepssh"
    assert len(call_order[0][2]) == 7
    assert call_order[0][2][0][5] == "18001:localhost:8000"
    assert call_order[0][2][1][5] == "18002:localhost:8000"


def test_start_everything_passes_routing_override_to_load_balancer(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(main.DEFAULT_CONFIG)
    seen = []

    monkeypatch.setattr(main, "launch_in_tmux", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "serve_forever",
        lambda path, verbose=False, routing=None: seen.append((path, verbose, routing)),
    )

    result = main.start_everything(config_path, routing="random")

    assert result == 0
    assert seen == [(config_path, False, "random")]


def test_start_everything_skips_tmux_for_direct_setup(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(main.DEFAULT_CONFIG.replace('- setup: "ssh"', '- setup: "direct"'))
    seen = []

    monkeypatch.setattr(main, "launch_in_tmux", lambda *args, **kwargs: seen.append("tmux"))
    monkeypatch.setattr(main, "serve_forever", lambda path, verbose=False: seen.append((path, verbose)))

    result = main.start_everything(config_path)

    assert result == 0
    assert seen == [(config_path, False)]


def test_start_everything_returns_failure_when_tmux_launch_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(main.DEFAULT_CONFIG)
    logs = []
    served = []

    monkeypatch.setattr(
        main,
        "launch_in_tmux",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["tmux", "new-session"])
        ),
    )
    monkeypatch.setattr(main, "serve_forever", lambda *args, **kwargs: served.append("serve"))
    monkeypatch.setattr(main.logger, "error", lambda message, *args: logs.append(message.format(*args)))

    result = main.start_everything(config_path)

    assert result == 1
    assert served == []
    assert any("Failed to start tmux command" in line for line in logs)


def test_start_everything_logs_config_path_and_table(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(main.DEFAULT_CONFIG)
    logs = []

    def fake_info(message, *args):
        logs.append(message.format(*args))

    monkeypatch.setattr(main.logger, "info", fake_info)
    monkeypatch.setattr(main, "launch_in_tmux", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "serve_forever", lambda *args, **kwargs: None)

    result = main.start_everything(config_path)

    assert result == 0
    assert any(f"Using config file: {config_path}" in line for line in logs)
    table_logs = [line for line in logs if line.startswith("Loaded config:\n+")]
    assert len(table_logs) == 1
    assert "config-path" in table_logs[0]
    assert str(config_path) in table_logs[0]
    assert "endpoints" in table_logs[0]
    assert "worker-41:8000" in table_logs[0]


def test_parse_config_supports_compact_endpoint_port_rows(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                "  - worker-[41,45,49,53-54,57,59]:8000",
                "port:",
                "  - 8001",
                "load-balancer:",
                "  workers: 1",
                "",
            ]
        )
    )
    cfg = keep_connection.parse_config(config_path)

    assert cfg.hosts == [
        "worker-41",
        "worker-45",
        "worker-49",
        "worker-53",
        "worker-54",
        "worker-57",
        "worker-59",
    ]
    assert cfg.remote_ports == [8000] * 7
    assert cfg.endpoint_setup == "ssh"
    assert cfg.port_start == 18000
    assert cfg.load_balancer_health_path == "/models"


def test_parse_config_supports_direct_endpoint_setup(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                "  - worker-[41,45]:8000",
                '  - setup: "direct"',
                "port:",
                "  - 8001",
                "",
            ]
        )
    )

    cfg = keep_connection.parse_config(config_path)

    assert cfg.hosts == ["worker-41", "worker-45"]
    assert cfg.remote_ports == [8000, 8000]
    assert cfg.endpoint_setup == "direct"


def test_main_returns_130_on_keyboard_interrupt(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    def fake_start_everything(path, verbose=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main(["--config", str(config_path)])

    assert result == 130


def test_llm_proxy_entrypoint_is_available():
    result = subprocess.run(
        ["uv", "run", "llm-proxy", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: llm-proxy" in result.stdout


def test_packaged_cli_module_help_is_available():
    result = subprocess.run(
        ["uv", "run", "python", "-m", "llm_loadbalancer.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_packaged_console_scripts_are_available():
    for command in (
        "collect-jsonl",
        "build-unique-sft",
        "annotate-full-conversations",
        "cat_sft_messages",
    ):
        result = subprocess.run(
            ["uv", "run", command, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (command, result.stderr)
        assert "usage:" in result.stdout.lower()
