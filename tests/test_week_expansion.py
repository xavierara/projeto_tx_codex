import pandas as pd

from data_utils import person_week_expansion


def test_person_week_expansion_labels():
    df = pd.DataFrame(
        {
            "duration_weeks_capped": [3],
            "feature_a": [1],
        }
    )
    expanded = person_week_expansion(df, ["feature_a"])
    assert len(expanded) == 3
    assert expanded["y"].tolist() == [0, 0, 1]
    assert expanded["t_week"].tolist() == [1, 2, 3]
