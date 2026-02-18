from __future__ import annotations

import polars as pl

from prism_ssl.data.filters import filter_likely_non_time_series


def test_filter_likely_non_time_series_removes_diffusion_and_cardiac() -> None:
    df = pl.DataFrame(
        {
            "series_description": [
                "CT ABDOMEN WITH CONTRAST",
                "MR DIFFUSION DWI",
                "CARDIAC CINE",
                "MR BRAIN T1",
            ],
            "exam_type": ["CT", "MR", "MR", "MR"],
        }
    )
    out = filter_likely_non_time_series(df)
    assert out["series_description"].to_list() == ["CT ABDOMEN WITH CONTRAST", "MR BRAIN T1"]


def test_filter_likely_non_time_series_noop_when_columns_missing() -> None:
    df = pl.DataFrame({"modality": ["CT", "MR"], "series_path": ["/a", "/b"]})
    out = filter_likely_non_time_series(df)
    assert out.equals(df)
