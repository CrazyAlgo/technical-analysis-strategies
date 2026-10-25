"""Tests for consolidated transaction-cost analysis."""

import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation import transaction_costs as tc
from technical_analysis.evaluation import metrics


@pytest.fixture
def sample_series():
    index = pd.RangeIndex(4)
    gross = pd.Series([0.02, -0.01, 0.03, 0.01], index=index)
    net = pd.Series([0.019, -0.011, 0.029, 0.009], index=index)
    costs = pd.Series([10.0, 5.0, 7.0, 8.0], index=index)
    return gross, net, costs


def test_analyze_transaction_costs_returns_expected_keys(sample_series):
    gross, net, costs = sample_series
    result = tc.analyze_transaction_costs(
        transaction_costs=costs,
        gross_returns=gross,
        net_returns=net,
    )
    assert set(result) == {"total_transaction_costs", "transaction_cost_drag"}


def test_total_transaction_costs_is_sum_of_cost_series(sample_series):
    gross, net, costs = sample_series
    result = tc.analyze_transaction_costs(costs, gross, net)
    assert result["total_transaction_costs"] == pytest.approx(costs.sum())


def test_transaction_cost_drag_matches_cumulative_return_difference(sample_series):
    gross, net, costs = sample_series
    result = tc.analyze_transaction_costs(costs, gross, net)
    expected = metrics.gross_return(gross) - metrics.net_return(net)
    assert result["transaction_cost_drag"] == pytest.approx(expected)


def test_zero_costs_are_reported_as_zero(sample_series):
    gross, net, _ = sample_series
    costs = pd.Series(0.0, index=gross.index)
    result = tc.analyze_transaction_costs(costs, gross, gross)
    assert result["total_transaction_costs"] == pytest.approx(0.0)
    assert result["transaction_cost_drag"] == pytest.approx(0.0)


def test_cost_analysis_rejects_misaligned_return_indices(sample_series):
    gross, net, costs = sample_series
    net.index = [10, 11, 12, 13]
    with pytest.raises((ValueError, TypeError)):
        tc.analyze_transaction_costs(costs, gross, net)


def test_cost_analysis_rejects_misaligned_cost_index(sample_series):
    gross, net, costs = sample_series
    costs.index = [10, 11, 12, 13]
    with pytest.raises((ValueError, TypeError)):
        tc.analyze_transaction_costs(costs, gross, net)


def test_cost_analysis_rejects_negative_costs(sample_series):
    gross, net, costs = sample_series
    costs.iloc[0] = -1.0
    with pytest.raises((ValueError, TypeError)):
        tc.analyze_transaction_costs(costs, gross, net)


def test_cost_analysis_rejects_nonfinite_costs(sample_series):
    gross, net, costs = sample_series
    costs.iloc[0] = np.inf
    with pytest.raises((ValueError, TypeError)):
        tc.analyze_transaction_costs(costs, gross, net)


def test_cost_analysis_rejects_empty_costs(sample_series):
    gross, net, _ = sample_series
    with pytest.raises((ValueError, TypeError)):
        tc.analyze_transaction_costs(pd.Series(dtype=float), gross, net)
