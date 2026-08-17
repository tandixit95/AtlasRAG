from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "verify_protocol.py"
SPEC = importlib.util.spec_from_file_location("reranker_protocol_verifier", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_protocol():
    return json.loads((MODULE_PATH.parent / "PROTOCOL.json").read_text())


def test_current_protocol_passes():
    assert MODULE.validate(valid_protocol()) == []


def test_final_dataset_payload_use_fails_closed():
    p = copy.deepcopy(valid_protocol())
    p["input_source"]["payload_from_final_evaluation_sets"] = True
    assert any("payload" in e for e in MODULE.validate(p))


def test_missing_frozen_dataset_prohibition_fails_closed():
    p = copy.deepcopy(valid_protocol())
    p["prohibited_dataset_names"] = ["scifact"]
    assert any("frozen final datasets" in e for e in MODULE.validate(p))


def test_default_promotion_claim_fails_closed():
    p = copy.deepcopy(valid_protocol())
    p["claim_boundary"]["default_promotion_allowed"] = True
    assert any("promotion claims" in e for e in MODULE.validate(p))
