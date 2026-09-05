"""Keep the public reviewer entry point executable rather than illustrative."""

import json
import subprocess
import sys

from atlasrag.demo import run_demo

EXPECTED = {
    "anonymous": ["memory://public"],
    "tenant_a_ops": ["memory://public", "memory://tenant-a"],
    "tenant_a_wrong_group": ["memory://public"],
}


def test_offline_demo_respects_tenant_and_group_boundaries():
    assert run_demo() == EXPECTED


def test_documented_script_runs_in_a_separate_interpreter():
    result = subprocess.run(
        [sys.executable, "-m", "atlasrag.demo"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert json.loads(result.stdout) == EXPECTED
