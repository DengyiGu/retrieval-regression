def test_plugin_passes_and_fails_with_useful_message(pytester):
    pytester.makepyfile("""
from retrieval_regression import Threshold

def test_pass(retrieval_check):
    result = retrieval_check({"q":{"d":1}}, {"q":{"d":1}}, {"q":{"d":1}},
                             k=[1], thresholds=[Threshold("recall@1", max_drop=0)])
    assert result.passed

def test_fail(retrieval_check):
    retrieval_check({"q":{"d":1}}, {"q":{"d":1}}, {},
                    k=[1], thresholds=[Threshold("recall@1", max_drop=0)])


def test_requires_gate(retrieval_check):
    import pytest
    with pytest.raises(ValueError, match="explicit threshold"):
        retrieval_check({"q":{"d":1}}, {}, {})
""")
    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*Retrieval quality gates failed*", "*recall@1*"])
