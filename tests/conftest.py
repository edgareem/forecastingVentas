import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "fecha": ["2026-01-01", "2026-01-02"],
            "ventas": [100, 120],
        }
    )