import numpy as np
import pytest

import cpd


def test_fpop_fit_predict_roundtrip_l2() -> None:
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    result = cpd.Fpop(min_segment_len=2).fit(values).predict(pen=1.0)
    assert result.breakpoints == [5, 10]
    assert result.change_points == [5]


def test_fpop_predict_requires_exactly_one_stopping_arg() -> None:
    detector = cpd.Fpop().fit(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="exactly one"):
        detector.predict()

    with pytest.raises(ValueError, match="exactly one"):
        detector.predict(pen=1.0, n_bkps=1)


def test_fpop_predict_before_fit_is_clear_error() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\.\\.\\.\\) must be called before predict"):
        cpd.Fpop().predict(pen=1.0)


def test_fpop_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="min_segment_len"):
        cpd.Fpop(min_segment_len=0)

    with pytest.raises(ValueError, match="jump"):
        cpd.Fpop(jump=0)
