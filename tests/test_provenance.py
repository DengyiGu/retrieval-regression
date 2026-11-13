import hashlib
import json

import pytest

from retrieval_regression import evaluate
from retrieval_regression.cli import main
from retrieval_regression.provenance import fingerprint, make_manifest, verify_inputs


def test_fingerprint_streams_full_content_and_detects_changes(tmp_path):
    path = tmp_path / "input.bin"
    data = b"abc" * 400000
    path.write_bytes(data)
    record = fingerprint("run", path, "json")
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    assert record.size_bytes == len(data)
    verify_inputs((record,))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs((record,))


def test_manifest_is_portable_and_result_hash_can_be_verified(tmp_path):
    path = tmp_path / "qrels.json"
    path.write_text('{"q":{"d":1}}', encoding="utf-8")
    result = evaluate({"q": {"d": 1}}, {}, k=[1])
    manifest = make_manifest(
        result, (fingerprint("qrels", path, "json"),), {"k": [1]}, version="test-version"
    )
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["inputs"][0]["name"] == "qrels.json"
    assert manifest["quality_gated"] is False
    canonical = json.dumps(
        result.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert manifest["result_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_cli_compare_manifest_matches_output_and_tracks_config(examples, tmp_path, capsys):
    manifest = tmp_path / "nested" / "manifest.json"
    result = main(
        [
            "compare",
            "--qrels",
            str(examples / "qrels.json"),
            "--baseline",
            str(examples / "baseline.json"),
            "--candidate",
            str(examples / "regressed.json"),
            "--config",
            str(examples / "gates.json"),
            "--k",
            "1",
            "3",
            "--manifest",
            str(manifest),
        ]
    )
    assert result == 1
    output = json.loads(capsys.readouterr().out)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["quality_gated"] is True
    assert {item["role"] for item in record["inputs"]} == {
        "qrels",
        "baseline",
        "candidate",
        "config",
    }
    assert record["parameters"]["seed"] == 0
    canonical = json.dumps(
        output, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert record["result_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_cli_evaluate_manifest_and_collision_protection(examples, tmp_path, capsys):
    qrels = examples / "qrels.json"
    run = examples / "baseline.json"
    target = tmp_path / "manifest.json"
    assert (
        main(["evaluate", "--qrels", str(qrels), "--run", str(run), "--manifest", str(target)]) == 0
    )
    assert json.loads(target.read_text())["parameters"]["command"] == "evaluate"
    assert (
        main(["evaluate", "--qrels", str(qrels), "--run", str(run), "--manifest", str(qrels)]) == 2
    )
    assert "overwrite input" in capsys.readouterr().err


def test_changed_input_aborts_before_creating_any_report(examples, tmp_path, monkeypatch, capsys):
    import retrieval_regression.cli as cli

    path = tmp_path / "run.json"
    path.write_text((examples / "baseline.json").read_text(), encoding="utf-8")
    original_loader = cli.load_run

    def changed_loader(source, **kwargs):
        result = original_loader(source, **kwargs)
        path.write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(cli, "load_run", changed_loader)
    report, manifest = tmp_path / "result.json", tmp_path / "manifest.json"
    assert (
        main(
            [
                "evaluate",
                "--qrels",
                str(examples / "qrels.json"),
                "--run",
                str(path),
                "--json",
                str(report),
                "--manifest",
                str(manifest),
            ]
        )
        == 2
    )
    assert "input changed" in capsys.readouterr().err
    assert not report.exists() and not manifest.exists()


def test_manifest_cannot_alias_an_existing_json_report(examples, tmp_path, capsys):
    import os

    output, manifest = tmp_path / "result.json", tmp_path / "manifest.json"
    output.write_text("previous report", encoding="utf-8")
    os.link(output, manifest)
    assert (
        main(
            [
                "evaluate",
                "--qrels",
                str(examples / "qrels.json"),
                "--run",
                str(examples / "baseline.json"),
                "--json",
                str(output),
                "--manifest",
                str(manifest),
            ]
        )
        == 2
    )
    assert "must be different" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "previous report"
