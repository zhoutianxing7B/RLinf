import pytest

from rlinf.utils.rollout_sampling import evenly_spaced_indices


def test_evenly_spaced_indices_cover_full_range():
    indices = evenly_spaced_indices(180, 20)
    assert len(indices) == 20
    assert len(set(indices)) == 20
    assert indices[0] == 0
    assert indices[-1] == 179


def test_evenly_spaced_indices_support_empty_and_singleton():
    assert evenly_spaced_indices(10, 0) == []
    assert evenly_spaced_indices(10, 1) == [0]


def test_evenly_spaced_indices_reject_oversampling():
    with pytest.raises(ValueError, match="cannot exceed"):
        evenly_spaced_indices(3, 4)
