"""Format contracts and boundary behavior, independent of parser implementation."""

import json
import subprocess
import sys

import pytest

from retrieval_regression import InputFormatError, compare, evaluate, load_qrels, load_run
from retrieval_regression.adapters import read_beir_qrels, read_trec_qrels, read_trec_run


def save(tmp_path, text, name="input.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_beir_preserves_ids_zeros_bom_and_interleaved_queries(tmp_path):
    path = save(
        tmp_path, "\ufeff\r\nquery-id\tcorpus-id\tscore\r\n001\ta\t2\r\nq2\tb\t1\r\n001\tc\t0\r\n"
    )
    assert read_beir_qrels(path) == {"001": {"a": 2, "c": 0}, "q2": {"b": 1}}


def test_trec_qrels_equivalent_to_beir(tmp_path):
    beir = save(tmp_path, "query-id\tcorpus-id\tscore\nq1\ta\t3\nq1\tb\t0\n", "qrels.tsv")
    trec = save(tmp_path, "q1 0 a 3\nq1 0 b 0\n", "qrels.trec")
    assert load_qrels(beir, format="beir") == load_qrels(trec, format="trec")


@pytest.mark.parametrize(
    "text,line",
    [
        ("", None),
        ("query-id corpus-id score\nq d 1", 1),
        ("query-id\tcorpus-id\twrong\nq\td\t1", 1),
        ("query-id\tcorpus-id\tscore\n", None),
        ("query-id\tcorpus-id\tscore\nq\td", 2),
        ("query-id\tcorpus-id\tscore\nq\td\t1\textra", 2),
        ("query-id\tcorpus-id\tscore\n\td\t1", 2),
        ("query-id\tcorpus-id\tscore\nq\twhite space\t1", 2),
        ("query-id\tcorpus-id\tscore\n q\td\t1", 2),
        ("query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t1", 3),
        ("query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t2", 3),
        ("query-id\tcorpus-id\tscore\nq\td\t0", 2),
        ("query-id\tcorpus-id\tscore\nq\td\t-1", 2),
        ("query-id\tcorpus-id\tscore\nq\td\t1.5", 2),
        ("query-id\tcorpus-id\tscore\nq\td\tNaN", 2),
        ("query-id\tcorpus-id\tscore\nq\td\t" + "9" * 400, 2),
    ],
)
def test_bad_beir_has_file_and_physical_line(tmp_path, text, line):
    path = save(tmp_path, text)
    with pytest.raises(InputFormatError) as error:
        read_beir_qrels(path)
    assert error.value.path == str(path)
    assert error.value.line_number == line
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    "text,line",
    [
        ("", None),
        ("q 0 d", 1),
        ("q Q0 d 1", 1),
        ("q 0 d -1", 1),
        ("q 0 d 1.0", 1),
        ("q 0 d 0\nq 0 other 0", 1),
        ("q 0 d 1\n\nq 0 d 2", 3),
    ],
)
def test_bad_trec_qrels_rejected(tmp_path, text, line):
    with pytest.raises(InputFormatError) as error:
        read_trec_qrels(save(tmp_path, text))
    assert error.value.line_number == line


def test_run_rows_can_be_interleaved_out_of_order_and_have_rank_gaps(tmp_path):
    path = save(tmp_path, "q Q0 b 8 -2 run\nother Q0 c 1 1e-3 run\nq Q0 a 2 0 run\n")
    assert read_trec_run(path) == {"q": {"a": 0, "b": -2}, "other": {"c": 0.001}}


def test_rank_metadata_does_not_override_document_id_score_ties(tmp_path):
    path = save(tmp_path, "q Q0 z 1 2 run\nq Q0 a 2 2 run\n")
    result = evaluate({"q": {"a": 1}}, load_run(path, format="trec"), k=[1])
    assert result.summary["recall@1"] == 1


@pytest.mark.parametrize(
    "text,line",
    [
        ("q Q0 d 1 1", 1),
        ("q Q0 d 1 1 run extra", 1),
        ("q 0 d 1 1 run", 1),
        ("q Q0 d 0 1 run", 1),
        ("q Q0 d -1 1 run", 1),
        ("q Q0 d 1.0 1 run", 1),
        ("q Q0 d 1 bad run", 1),
        ("q Q0 d 1 NaN run", 1),
        ("q Q0 d 1 Inf run", 1),
        ("q Q0 d 1 1e999 run", 1),
        ("q Q0 d 1 1 run\nq Q0 d 2 0 run", 2),
        ("q Q0 a 1 1 run\nq Q0 b 1 0 run", 2),
        ("q Q0 a 1 1 run\nother Q0 b 1 1 different", 2),
        ("q Q0 a 1 1 run\n\nq Q0 b 2 2 run", 3),
    ],
)
def test_invalid_runs_fail_with_location(tmp_path, text, line):
    with pytest.raises(InputFormatError) as error:
        read_trec_run(save(tmp_path, text))
    assert error.value.line_number == line


def test_empty_run_retains_all_queries_and_unknown_queries_are_rejected(tmp_path):
    empty = load_run(save(tmp_path, "\n"), format="trec")
    result = compare({"q": {"d": 1}}, {"q": {"d": 1}}, empty, k=[1])
    assert result.candidate.query_count == 1
    assert result.delta["ndcg@1"] == -1
    unknown = load_run(save(tmp_path, "typo Q0 d 1 1 run"), format="trec")
    with pytest.raises(ValueError, match="unknown query"):
        evaluate({"q": {"d": 1}}, unknown)


def test_json_remains_default_and_fractional_grades_remain_supported(tmp_path):
    path = save(tmp_path, '{"q":{"d":0.5}}')
    assert load_qrels(path) == {"q": {"d": 0.5}}
    assert load_run(path) == {"q": {"d": 0.5}}
    with pytest.raises(ValueError, match="unknown qrels format"):
        load_qrels(path, format="csv")
    with pytest.raises(ValueError, match="unknown run format"):
        load_run(path, format="csv")


def test_cli_mixed_formats_and_errors_do_not_create_outputs(tmp_path):
    qrels = save(tmp_path, "query-id\tcorpus-id\tscore\nq\ta\t1\n", "qrels.tsv")
    base = save(tmp_path, "q Q0 a 1 2 baseline\n", "base.run")
    candidate = save(tmp_path, '{"q":{"a":1}}', "candidate.json")
    config = save(tmp_path, '{"thresholds":[{"metric":"ndcg@1","max_drop":0}]}', "gates.json")
    command = [
        sys.executable,
        "-m",
        "retrieval_regression",
        "compare",
        "--qrels",
        str(qrels),
        "--qrels-format",
        "beir",
        "--baseline",
        str(base),
        "--baseline-format",
        "trec",
        "--candidate",
        str(candidate),
        "--config",
        str(config),
        "--k",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"]
    candidate.write_text("q Q0 a 1 1 broken\nq Q0 a 2 0 broken", encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [*command, "--candidate-format", "trec", "--json", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert f"{candidate}:2:" in result.stderr
    assert not output.exists()


def test_cli_evaluate_trec_qrels_and_run(tmp_path):
    qrels = save(tmp_path, "q 0 d 1", "qrels.txt")
    run = save(tmp_path, "q Q0 d 1 1 run", "run.txt")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "retrieval_regression",
            "evaluate",
            "--qrels",
            str(qrels),
            "--qrels-format",
            "trec",
            "--run",
            str(run),
            "--run-format",
            "trec",
            "--k",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["summary"]["ndcg@1"] == 1


@pytest.mark.parametrize("reader", [read_beir_qrels, read_trec_qrels, read_trec_run])
def test_invalid_encoding_has_a_file_level_error(tmp_path, reader):
    path = tmp_path / "bad-encoding.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(InputFormatError, match="UTF-8") as error:
        reader(path)
    assert error.value.path == str(path)
    assert error.value.line_number is None
