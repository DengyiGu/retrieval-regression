"""Decisions are checked against known distributions, not the implementation formula."""

import json

import pytest

from retrieval_regression import Threshold, compare, evaluate, load_qrels
from retrieval_regression.cli import main
from retrieval_regression.pytest_plugin import RetrievalCheck
from retrieval_regression.reporting import write_html


def runs(n=40):
    qrels = {str(i): {"a": 1} for i in range(n)}
    good = {q: {"a": 1} for q in qrels}
    bad = {q: {"z": 1} for q in qrels}
    mixed = {q: good[q] if int(q) % 2 else bad[q] for q in qrels}
    return qrels, good, bad, mixed


def gated(qrels, base, cand, **kwargs):
    return compare(
        qrels,
        base,
        cand,
        k=[1],
        gate_mode="bootstrap",
        bootstrap_samples=1000,
        thresholds=[Threshold("ndcg@1", max_drop=0.02)],
        **kwargs,
    )


def test_interval_gates_distinguish_pass_fail_and_inconclusive():
    q, good, bad, mixed = runs()
    assert gated(q, good, good).status == "pass"
    assert gated(q, good, bad).status == "fail"
    inverse = {qid: bad[qid] if int(qid) % 2 else good[qid] for qid in q}
    result = gated(q, mixed, inverse)
    assert result.status == "inconclusive" and not result.passed
    ci = result.gates[0].evidence["max_drop"]
    assert ci.low < -0.02 < ci.high
    assert result.to_dict()["status"] == "inconclusive"


def test_small_groups_get_their_own_intervals_and_sample_guard():
    q, good, bad, _ = runs()
    groups = {qid: "small" if int(qid) < 3 else "large" for qid in q}
    result = compare(
        q,
        good,
        bad,
        k=[1],
        groups=groups,
        bootstrap_samples=1000,
        gate_mode="bootstrap",
        thresholds=[Threshold("ndcg@1", max_drop=0.02, group="small")],
    )
    assert result.gates[0].query_count == 3
    assert result.gates[0].status == "inconclusive"
    assert result.groups["small"]["confidence_intervals"]["ndcg@1"]["high"] == -1
    assert "min_queries" in result.gates[0].reason


def test_minimum_and_drop_use_different_evidence_and_adjustment():
    q, good, _, mixed = runs()
    result = compare(
        q,
        good,
        mixed,
        k=[1],
        bootstrap_samples=1000,
        gate_mode="bootstrap",
        thresholds=[Threshold("ndcg@1", min_value=0.2, max_drop=0.1)],
    )
    gate = result.gates[0]
    assert gate.status == "fail"
    assert gate.evidence["min_value"].low > 0.2
    assert gate.evidence["max_drop"].high < -0.1
    assert gate.evidence["min_value"].confidence == pytest.approx(0.975)
    unadjusted = compare(
        q,
        good,
        mixed,
        k=[1],
        bootstrap_samples=1000,
        gate_mode="bootstrap",
        multiple_testing="none",
        thresholds=[Threshold("ndcg@1", min_value=0.2, max_drop=0.1)],
    )
    assert unadjusted.gates[0].evidence["min_value"].confidence == 0.95


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_mode": "typo"},
        {"gate_mode": "bootstrap"},
        {"min_queries": 1},
        {"multiple_testing": "typo"},
    ],
)
def test_invalid_gate_configuration(kwargs):
    q, good, _, _ = runs()
    with pytest.raises(ValueError):
        compare(q, good, good, **kwargs)


@pytest.mark.parametrize(
    "fmt,content",
    [
        ("json", '{"q":{"a":1,"z":-1},"empty":{"z":0}}'),
        ("beir", "query-id\tcorpus-id\tscore\nq\ta\t1\nq\tz\t-1\nempty\tz\t0\n"),
        ("trec", "q 0 a 1\nq 0 z -1\nempty 0 z 0\n"),
    ],
)
def test_explicit_qrels_policies_preserve_auditable_denominators(tmp_path, fmt, content):
    path = tmp_path / "judgments"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_qrels(path, format=fmt)
    q = load_qrels(path, format=fmt, strict=False)
    run = {"q": {"a": 1}, "empty": {"z": 1}}
    kept = evaluate(q, run, k=[1], negative_grades="clip", no_positive="keep")
    assert kept.summary["ndcg@1"] == 0.5
    assert kept.query_count == kept.input_query_count == 2
    assert {d.code for d in kept.diagnostics} >= {"negative_grade_clip", "no_positive_keep"}
    dropped = compare(
        q,
        run,
        run,
        k=[1],
        negative_grades="drop",
        no_positive="drop",
        groups={"empty": "excluded", "q": "included"},
    )
    assert dropped.baseline.summary["ndcg@1"] == 1
    assert dropped.baseline.query_count == 1 and dropped.baseline.input_query_count == 2
    assert set(dropped.groups) == {"included"}
    with pytest.raises(ValueError, match="unknown query"):
        evaluate(q, {"alien": {}}, negative_grades="drop", no_positive="drop")


@pytest.mark.parametrize(
    "kwargs", [{"negative_grades": "typo"}, {"no_positive": "typo"}, {"tie_breaker": "typo"}]
)
def test_invalid_policies(kwargs):
    with pytest.raises(ValueError):
        evaluate({"q": {"a": 1}}, {}, **kwargs)


def test_all_dropped_and_negative_only_kept_queries():
    with pytest.raises(ValueError, match="every query"):
        evaluate({"q": {"a": 0}}, {}, no_positive="drop")
    kept = evaluate({"q": {"a": -1}}, {}, negative_grades="drop", no_positive="keep")
    assert all(v == 0 for v in kept.summary.values())


def test_tie_order_is_explicit_and_default_is_preserved():
    q, run = {"q": {"z": 1}}, {"q": {"a": 2, "z": 2}}
    assert evaluate(q, run, k=[1]).summary["ndcg@1"] == 0
    assert evaluate(q, run, k=[1], tie_breaker="doc_id_desc").summary["ndcg@1"] == 1


def test_cli_summary_retains_full_file_and_inconclusive_exit_policy(tmp_path, capsys):
    q, good, _, _ = runs(3)
    paths = []
    for name, data in [
        ("qrels", q),
        ("run", good),
        ("gates", {"thresholds": [{"metric": "ndcg@1", "max_drop": 0.02}]}),
    ]:
        path = tmp_path / name
        path.write_text(json.dumps(data))
        paths.append(str(path))
    output = tmp_path / "full.json"
    args = [
        "compare",
        "--qrels",
        paths[0],
        "--baseline",
        paths[1],
        "--candidate",
        paths[1],
        "--config",
        paths[2],
        "--k",
        "1",
        "--gate-mode",
        "bootstrap",
        "--bootstrap-samples",
        "1000",
        "--json",
        str(output),
        "--summary",
    ]
    assert main(args) == 3
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "inconclusive" and "per_query_delta" not in summary
    assert "per_query_delta" in json.loads(output.read_text())
    assert main(args + ["--on-inconclusive", "warn"]) == 0
    capsys.readouterr()
    assert main(args + ["--stdout", "none"]) == 3
    assert capsys.readouterr().out == ""
    assert main(["evaluate", "--qrels", paths[0], "--run", paths[1], "--summary"]) == 0
    assert "per_query" not in json.loads(capsys.readouterr().out)


def test_html_and_pytest_expose_inconclusive(tmp_path):
    q, good, _, _ = runs(3)
    result = gated(q, good, good)
    report = tmp_path / "report.html"
    write_html(result, report)
    assert "INCONCLUSIVE" in report.read_text()
    assert "interval [" in report.read_text()
    check = RetrievalCheck()
    kwargs = dict(
        k=[1],
        gate_mode="bootstrap",
        bootstrap_samples=1000,
        thresholds=[Threshold("ndcg@1", max_drop=0.02)],
    )
    with pytest.raises(pytest.fail.Exception, match="INCONCLUSIVE"):
        check(q, good, good, **kwargs)
    with pytest.warns(UserWarning, match="inconclusive"):
        assert check(q, good, good, on_inconclusive="warn", **kwargs).status == "inconclusive"
    with pytest.raises(ValueError, match="on_inconclusive"):
        check(q, good, good, on_inconclusive="typo", **kwargs)
