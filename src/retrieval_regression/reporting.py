"""Portable reports without JavaScript, network resources, or model calls."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .models import ComparisonResult, EvaluationResult


def write_json(result: ComparisonResult | EvaluationResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_html(result: ComparisonResult, path: str | Path) -> None:
    escape = html.escape
    rows = "".join(
        f"<tr><th>{escape(metric)}</th><td>{result.baseline.summary[metric]:.6f}</td>"
        f"<td>{result.candidate.summary[metric]:.6f}</td><td>{delta:+.6f}</td></tr>"
        for metric, delta in result.delta.items()
    )
    focus = f"ndcg@{max(result.baseline.k)}"
    query_rows = "".join(
        f"<tr><th>{escape(qid)}</th><td>{result.baseline.per_query[qid][focus]:.6f}</td>"
        f"<td>{result.candidate.per_query[qid][focus]:.6f}</td>"
        f"<td>{result.per_query_delta[qid][focus]:+.6f}</td></tr>"
        for qid in sorted(
            result.per_query_delta, key=lambda q: (result.per_query_delta[q][focus], q)
        )
    )
    diagnostics = (
        "".join(
            f"<li>{label}: {escape(item.query_id)}: {escape(item.message)}</li>"
            for label, items in (
                ("Baseline", result.baseline.diagnostics),
                ("Candidate", result.candidate.diagnostics),
            )
            for item in items
        )
        or "<li>No input diagnostics.</li>"
    )
    gate_items = (
        "".join(
            f"<li>{gate.status.upper()}: {escape(gate.metric)} "
            f"({escape(gate.group or 'overall')}), min={gate.min_value}, "
            f"max absolute drop={gate.max_drop}; n={gate.query_count}. {escape(gate.reason)}"
            + "".join(
                f"; {escape(name)} interval [{ci.low:+.6f}, {ci.high:+.6f}] ({ci.confidence:.2%})"
                for name, ci in gate.evidence.items()
            )
            + "</li>"
            for gate in result.gates
        )
        or "<li>No thresholds configured; quality was not gated.</li>"
    )
    group_sections = "".join(
        f"<h3>{escape(group)} ({values['query_count']} queries)</h3><ul>"
        + "".join(
            f"<li>{escape(metric)}: {change:+.6f}"
            + (
                f" [{values['confidence_intervals'][metric]['low']:+.6f}, "
                f"{values['confidence_intervals'][metric]['high']:+.6f}]"
                if "confidence_intervals" in values
                else ""
            )
            + "</li>"
            for metric, change in values["delta"].items()
        )
        + "</ul>"
        for group, values in result.groups.items()
    )
    intervals = (
        "".join(
            f"<li>{escape(metric)}: [{ci.low:+.6f}, {ci.high:+.6f}] "
            f"({ci.confidence:.0%}, {ci.samples} paired bootstrap samples)</li>"
            for metric, ci in result.confidence_intervals.items()
        )
        or "<li>Bootstrap intervals not requested.</li>"
    )
    status = result.status.upper() if result.gates else "NOT CONFIGURED"
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Retrieval Regression Report</title><style>
body{{font:16px system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;
color:#172033}}
h1{{color:#164e63}}strong{{background:#e0f2fe;padding:5px 10px;border-radius:6px}}
h2{{margin-top:32px}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #ccd2dc;
padding:10px;text-align:left}}thead{{background:#eef2f8}}code{{background:#eef2f8;padding:2px}}
</style></head><body><h1>Retrieval Regression</h1>
<p>Gate status: <strong>{status}</strong>. Evaluated queries: {result.baseline.query_count}.</p>
<p>Delta = candidate minus baseline. Every eligible qrels query is included;
missing rankings score zero.</p>
<h2>Overall metrics</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th>
<th>Delta</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Quality gates ({escape(result.gate_mode)})</h2><ul>{gate_items}</ul>
<h2>Query changes ({escape(focus)}, worst first)</h2><table><thead><tr><th>Query</th>
<th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead><tbody>{query_rows}</tbody></table>
<h2>Category changes</h2>{group_sections or "<p>No categories supplied.</p>"}
<h2>Descriptive confidence intervals</h2><ul>{intervals}</ul>
<p>Descriptive intervals above are unadjusted. Gate evidence uses the configured
{escape(result.multiple_testing)} correction across configured bounds. Percentile bootstrap
coverage is approximate and assumes independent queries. Evidence can be inconclusive.</p>
<h2>Input diagnostics</h2><ul>{diagnostics}</ul>
<p>Unjudged documents count as non-relevant. Incomplete judgments limit interpretation.</p>
</body></html>
"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
