from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def examples() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "toy"
