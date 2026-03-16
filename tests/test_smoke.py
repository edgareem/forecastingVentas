from src.forecasting_ventas.features.build_features import build_features


def test_build_features_returns_copy(sample_dataframe):
    result = build_features(sample_dataframe)

    assert list(result.columns) == list(sample_dataframe.columns)
    assert result is not sample_dataframe