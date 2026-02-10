import pandas as pd

from data_utils import add_time_features


def test_duration_weeks_capped():
    df = pd.DataFrame(
        {
            "dateCreated": ["2020-01-01"],
            "lastSeen": ["2020-02-15"],
        }
    )
    enriched = add_time_features(df, t_max=4)
    assert enriched.loc[0, "duration_weeks"] >= 1
    assert enriched.loc[0, "duration_weeks_capped"] == 4
