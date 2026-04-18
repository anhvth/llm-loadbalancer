import pathlib
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

    def fake_start_everything(path, verbose=False):
        seen.append((path, verbose))
        return 0

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main([])

    assert result == 0
    assert seen == [(tmp_path / "config.yaml", True)]


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
    assert len(call_order[0][2]) == 8
    assert call_order[0][2][0][:4] == [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
    ]


def test_main_returns_130_on_keyboard_interrupt(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"

    def fake_start_everything(path, verbose=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "start_everything", fake_start_everything)

    result = main.main(["--config", str(config_path)])

    assert result == 130


def write_log_dir(log_dir: pathlib.Path, rows: list[tuple[str, str, str]]):
    requests_dir = log_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    for index, (input_text, output_text, endpoint_used) in enumerate(rows, start=1):
        (requests_dir / f"{index:06d}.json").write_text(
            json.dumps(
                {
                    "input": input_text,
                    "output": output_text,
                    "endpoint_used": endpoint_used,
                }
            ),
            encoding="utf-8",
        )


def test_cat_db_uses_default_cache_path(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "cache-home" / ".cache" / "llmup" / "logs"
    monkeypatch.setenv("HOME", str(tmp_path / "cache-home"))
    write_log_dir(
        db_path,
        [('{"a":1}', '{"b":2}', "http://127.0.0.1:18000/v1/chat/completions")],
    )

    result = cat_db.main([])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert "Row 1" in captured.out
    assert "endpoint_used: http://127.0.0.1:18000/v1/chat/completions" in captured.out
    assert "input:" in captured.out
    assert '"a": 1' in captured.out
    assert "output:" in captured.out
    assert '"b": 2' in captured.out


def test_cat_db_accepts_explicit_db_path(tmp_path, capsys):
    db_path = tmp_path / "custom-logs"
    write_log_dir(
        db_path,
        [
            ('{"a":1}', '{"b":2}', "http://127.0.0.1:18000/v1/chat/completions"),
            ('{"a":2}', '{"b":3}', "http://127.0.0.1:18001/v1/chat/completions"),
        ],
    )

    result = cat_db.main(["--raw", str(db_path)])

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


def test_cat_db_render_row_pretty_formats_messages_readably():
    row = (
        7,
        json.dumps(
            {
                "model": "demo",
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello"},
                ],
            }
        ),
        json.dumps({"id": "resp_1", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}),
        "http://127.0.0.1:18000/v1/chat/completions",
    )

    rendered = cat_db.render_row_pretty(row)

    assert "Row 7" in rendered
    assert "input:" in rendered
    assert "messages:" in rendered
    assert "[1] system" in rendered
    assert "content: be concise" in rendered
    assert "[2] user" in rendered
    assert "content: hello" in rendered
    assert "output:" in rendered
    assert '"id": "resp_1"' in rendered


def test_cat_db_uses_non_interactive_output_when_stdout_is_not_a_tty(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "custom-logs"
    write_log_dir(
        db_path,
        [('{"a":1}', '{"b":2}', "http://127.0.0.1:18000/v1/chat/completions")],
    )
    monkeypatch.setattr(cat_db.sys.stdout, "isatty", lambda: False)

    result = cat_db.main([str(db_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert "Navigation" not in captured.out
    assert "Row 1" in captured.out


class _FakeStdin:
    def __init__(self, reads):
        self._reads = list(reads)

    def fileno(self):
        return 0

    def read(self, _size):
        result = self._reads.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_cat_db_read_key_maps_uppercase_g_to_end(monkeypatch):
    monkeypatch.setattr(cat_db, "termios", type("FakeTermios", (), {"tcgetattr": staticmethod(lambda fd: "state"), "tcsetattr": staticmethod(lambda fd, when, state: None), "TCSADRAIN": object()})())
    monkeypatch.setattr(cat_db, "tty", type("FakeTty", (), {"setraw": staticmethod(lambda fd: None)})())
    monkeypatch.setattr(cat_db.sys, "stdin", _FakeStdin(["G"]))

    assert cat_db.read_key() == "end"


def test_cat_db_read_key_maps_double_g_to_home(monkeypatch):
    monkeypatch.setattr(cat_db, "termios", type("FakeTermios", (), {"tcgetattr": staticmethod(lambda fd: "state"), "tcsetattr": staticmethod(lambda fd, when, state: None), "TCSADRAIN": object()})())
    monkeypatch.setattr(cat_db, "tty", type("FakeTty", (), {"setraw": staticmethod(lambda fd: None)})())
    monkeypatch.setattr(cat_db.sys, "stdin", _FakeStdin(["g", "g"]))
    monkeypatch.setattr(cat_db.fcntl, "fcntl", lambda fd, op, arg=None: 0)

    assert cat_db.read_key() == "home"


def test_cat_db_read_key_handles_single_g_without_crashing(monkeypatch):
    monkeypatch.setattr(cat_db, "termios", type("FakeTermios", (), {"tcgetattr": staticmethod(lambda fd: "state"), "tcsetattr": staticmethod(lambda fd, when, state: None), "TCSADRAIN": object()})())
    monkeypatch.setattr(cat_db, "tty", type("FakeTty", (), {"setraw": staticmethod(lambda fd: None)})())
    monkeypatch.setattr(cat_db.sys, "stdin", _FakeStdin(["g", BlockingIOError()]))
    monkeypatch.setattr(cat_db.fcntl, "fcntl", lambda fd, op, arg=None: 0)

    assert cat_db.read_key() == "g"
