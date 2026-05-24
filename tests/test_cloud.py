"""Tests for the cloud subpackage that don't require live Lambda credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from arguenaut.cloud.lambda_api import LambdaCloudClient, LambdaCloudError
from arguenaut.cloud.provisioner import _sh_quote
from arguenaut.cloud.state import LambdaState, clear_state, load_state, save_state


def test_state_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARGUENAUT_LAMBDA_STATE", str(tmp_path / "s.json"))
    assert load_state() is None

    s = LambdaState(
        instance_id="i-abc",
        name="arguenaut",
        ip="1.2.3.4",
        ssh_user="ubuntu",
        api_url="http://1.2.3.4:8000",
        api_port=8000,
        region="us-west-1",
        instance_type="gpu_1x_a10",
        launched_at=123.0,
    )
    save_state(s)
    got = load_state()
    assert got is not None
    assert got.instance_id == "i-abc"
    assert got.api_url == "http://1.2.3.4:8000"
    assert got.bootstrap_complete is False

    clear_state()
    assert load_state() is None


def test_state_corrupted_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARGUENAUT_LAMBDA_STATE", str(tmp_path / "s.json"))
    (tmp_path / "s.json").write_text("{not valid json")
    assert load_state() is None  # corrupt → treated as missing


def test_sh_quote():
    assert _sh_quote("hello") == "'hello'"
    assert _sh_quote("it's") == "'it'\\''s'"
    assert _sh_quote("") == "''"
    assert _sh_quote(None) == "''"


def test_lambda_client_requires_key():
    with pytest.raises(LambdaCloudError):
        LambdaCloudClient("")
