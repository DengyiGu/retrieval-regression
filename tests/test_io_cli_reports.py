import json
import subprocess
import sys

import pytest

from retrieval_regression import compare
from retrieval_regression.io import load_json, load_thresholds
from retrieval_regression.reporting import write_html, write_json


@pytest.mark.parametrize(
    "text", ['{"q":{"d":1,"d":2}}', '{"q":{},"q":{}}', '{"q":{"d":NaN}}', '{"q":{"d":Infinity}}']
)
def test_duplicate_keys_and_nonfinite_json_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_json(path)


def test_bom_supported(tmp_path):
    path = tmp_path / "bom.json"
    path.write_text('{"q":{}}', encoding="utf-8-sig")
    assert load_json(path) == {"q": {}}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"thresholds": {}, "typo": 1},
        {"thresholds": [{"metric": "recall@1", "max_dorp": 0}]},
        {"thresholds": [{"metric": 1, "max_drop": 0}]},
        {"thresholds": [{"metric": "recall@1", "group": [], "max_drop": 0}]},
    ],
)
def test_bad_config_rejected(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)


def test_html_escapes_user_content_and_json_is_roundtrippable(tmp_path):
    query = '<script>alert("x")</script>'
    result = compare({query: {"d": 1}}, {}, {}, k=[1], groups={query: "<img src=x>"})
    report = tmp_path / "nested" / "report.html"
    write_html(result, report)
    text = report.read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "<img src=x>" not in text
    output = tmp_path / "result.json"
    write_json(result, output)
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "retrieval_regression", *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_cli_comparison_and_both_gate_exit_codes(examples, tmp_path):
    common = [
        "compare",
        "--qrels",
        examples / "qrels.json",
        "--baseline",
        examples / "baseline.json",
        "--k",
        "1",
        "3",
        "--config",
        examples / "gates.json",
    ]
    good = run_cli(
        *common,
        "--candidate",
        examples / "candidate.json",
        "--groups",
        examples / "groups.json",
        "--json",
        tmp_path / "result.json",
        "--html",
        tmp_path / "report.html",
    )
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["passed"]
    assert (tmp_path / "report.html").exists()
    bad = run_cli(*common, "--candidate", examples / "regressed.json")
    assert bad.returncode == 1, bad.stderr
    assert not json.loads(bad.stdout)["passed"]


def test_cli_invalid_input_has_exit_two_and_no_traceback(tmp_path):
    result = run_cli("evaluate", "--qrels", tmp_path / "missing.json", "--run", "missing.json")
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_evaluate_and_version(examples):
    result = run_cli(
        "evaluate",
        "--qrels",
        examples / "qrels.json",
        "--run",
        examples / "baseline.json",
        "--k",
        "1",
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["query_count"] == 4
    assert run_cli("--version").stdout.strip() == "0.1.0"


def test_cli_cannot_overwrite_its_inputs(examples, tmp_path):
    target = tmp_path / "qrels.json"
    original = (examples / "qrels.json").read_text(encoding="utf-8")
    target.write_text(original, encoding="utf-8")
    result = run_cli(
        "evaluate", "--qrels", target, "--run", examples / "baseline.json", "--json", target
    )
    assert result.returncode == 2
    assert "overwrite input" in result.stderr
    assert target.read_text(encoding="utf-8") == original


def test_cli_requires_distinct_report_paths(examples, tmp_path):
    result = run_cli(
        "compare",
        "--qrels",
        examples / "qrels.json",
        "--baseline",
        examples / "baseline.json",
        "--candidate",
        examples / "candidate.json",
        "--json",
        tmp_path / "same",
        "--html",
        tmp_path / "same",
    )
    assert result.returncode == 2
    assert "must be different" in result.stderr
