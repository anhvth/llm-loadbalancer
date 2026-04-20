from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from llm_loadbalancer import cli as packaged_cli
from llm_loadbalancer import keep_connection as packaged_keep_connection
from llm_loadbalancer import load_balancer as packaged_load_balancer


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_packaged_runtime_modules_import():
    assert packaged_cli.main is not None
    assert packaged_keep_connection.parse_config is not None
    assert packaged_load_balancer.create_app is not None


def test_packaged_runtime_modules_support_module_help():
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT / "src")
        if not pythonpath
        else os.pathsep.join([str(REPO_ROOT / "src"), pythonpath])
    )

    for module_name in (
        "llm_loadbalancer.cli",
        "llm_loadbalancer.keep_connection",
        "llm_loadbalancer.load_balancer",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
