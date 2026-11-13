# Development

## Local setup (Windows PowerShell)

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,reference]"
.\.venv\Scripts\python.exe scripts/check.py
.\.venv\Scripts\python.exe examples/quickstart.py
```

On POSIX use `.venv/bin/python`. No global installs or PowerShell execution-policy changes
are necessary. The base package has no runtime dependencies; pytest and reference tools are extras.
Use Python 3.10+; local validation currently records its actual interpreter in the validation report.

## Quality checks

`scripts/check.py` runs formatting, lint, strict source typing, coverage-based tests, and the
public pytest example using its own interpreter. The reference test skips visibly when
`ir-measures` is absent. CI installs it in its dedicated reference job.

To run checks individually:

```console
python -m ruff check .
python -m ruff format --check .
python -m mypy src/retrieval_regression
python -m coverage run -m pytest -q
python -m coverage combine
python -m coverage report
python -m pytest examples/test_retrieval_quality.py -q
```

Coverage starts before pytest plugin discovery and instruments Python subprocesses, including
CLI/plugin integration checks. Delete no unrelated work to reset a test run; use coverage's
own `erase` command when starting a fresh local coverage measurement.

## Packaging verification

```console
python -m build
python scripts/check_sdist.py dist
python -m twine check dist/*
python scripts/smoke_wheel.py dist --with-pytest
```

Validate the built wheel in a fresh environment with no editable checkout on `sys.path`,
then exercise `--version`, CLI comparison, and optional pytest plugin discovery. Build success
is not a published release. Publishing to PyPI or creating a release is a separate action.

## Implementing a task

1. Read the requirements, API contract, and evaluation conventions.
2. Choose a bounded roadmap item and define its observable acceptance behavior.
3. Implement in the appropriate module; avoid coupling the evaluator to integrations.
4. Add meaningful correctness/error-path tests and update affected examples/docs.
5. Run checks; record exact evidence and remaining limitations in the validation report.

## Layout and compatibility

Use `src/` layout and typed public interfaces. Keep user-facing docs in English.
Runtime core supports Python 3.10+. Metric semantics, gate meaning, CLI exit codes, and JSON
schema are compatibility surfaces. Do not silently change them to make a benchmark pass.

## Generated artifacts

`work/`, `reports/`, `data/`, `.venv/`, caches, coverage output, and distributions are ignored.
Commit only deliberate research summaries and compact derived evidence under `docs/reports/`.
Never commit external raw text corpora or environment files containing secrets/local credentials.

Build into a clean output directory: directory-based checks require exactly one matching artifact.
The check runner directs child-process temporary files to ignored `work/tmp/`; it does not change
system-wide TEMP settings.
