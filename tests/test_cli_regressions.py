import pathlib
import sqlite3
import json

import cat_db
import main


def test_default_config_path_uses_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert main.default_config_path() == tmp_path / "config.yaml"


def test_ensure_config_exists_creates_expected_default_file(tmp_path):
    config_path = tmp_path / "config.yaml"

    main.ensure_config_exists(config_path)

    assert config_path.read_text() == main.DEFAULT_CONFIG


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


def test_main_defaults_to_current_directory_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_start_everything(path):
        seen.append(path)
        return 0

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main([])

    assert result == 0
    assert seen == [tmp_path / "config.yaml"]


def test_main_passes_verbose_flag_to_start_everything(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    seen = []

    def fake_start_everything(path, verbose=False):
        seen.append((path, verbose))
        return 0

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main(["--config", str(config_path), "--verbose"])

    assert result == 0
    assert seen == [(config_path, True)]


def test_start_everything_launches_tmux_before_load_balancer(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(main.DEFAULT_CONFIG)
    call_order = []

    def fake_launch_in_tmux(session_name, commands):
        call_order.append(("tmux", session_name, commands))

    def fake_serve_forever(path):
        call_order.append(("serve", path))

    monkeypatch.setattr(main, "launch_in_tmux", fake_launch_in_tmux)
    monkeypatch.setattr(main, "serve_forever", fake_serve_forever)

    result = main.start_everything(config_path)

    assert result == 0
    assert call_order[0][0] == "tmux"
    assert call_order[1] == ("serve", config_path)
    assert call_order[0][1] == "keepssh"
    assert len(call_order[0][2]) == 8
    assert call_order[0][2][0][:4] == [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
    ]


def test_main_returns_130_on_keyboard_interrupt(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    def fake_start_everything(path):
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main(["--config", str(config_path)])

    assert result == 130


def write_log_db(db_path: pathlib.Path, rows: list[tuple[str, str, str]]):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE request_response_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                endpoint_used TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO request_response_log (input, output, endpoint_used) VALUES (?, ?, ?)",
            rows,
        )
        connection.commit()


def test_cat_db_uses_default_path_in_current_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "llm_loadbalancer.sqlite3"
    write_log_db(
        db_path,
        [('{"a":1}', '{"b":2}', "http://127.0.0.1:18000/v1/chat/completions")],
    )

    result = cat_db.main([])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out == (
        json.dumps(
            {
                "id": 1,
                "input": '{"a":1}',
                "output": '{"b":2}',
                "endpoint_used": "http://127.0.0.1:18000/v1/chat/completions",
            }
        )
        + "\n"
    )


def test_cat_db_accepts_explicit_db_path(tmp_path, capsys):
    db_path = tmp_path / "custom.sqlite3"
    write_log_db(
        db_path,
        [
            ('{"a":1}', '{"b":2}', "http://127.0.0.1:18000/v1/chat/completions"),
            ('{"a":2}', '{"b":3}', "http://127.0.0.1:18001/v1/chat/completions"),
        ],
    )

    result = cat_db.main([str(db_path)])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert result == 0
    assert captured.err == ""
    assert [json.loads(line) for line in lines] == [
        {
            "id": 1,
            "input": '{"a":1}',
            "output": '{"b":2}',
            "endpoint_used": "http://127.0.0.1:18000/v1/chat/completions",
        },
        {
            "id": 2,
            "input": '{"a":2}',
            "output": '{"b":3}',
            "endpoint_used": "http://127.0.0.1:18001/v1/chat/completions",
        },
    ]
