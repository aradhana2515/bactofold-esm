import pandas as pd

from bactofold.chaperone import (
    add_chaperone_delta_columns,
    add_chaperone_rescue_labels,
)


def test_chaperone_rescue_labels():
    df = pd.DataFrame(
        {
            "minus": [10.0, 25.0],
            "TF": [35.0, 30.0],
            "GroE": [15.0, 60.0],
            "KJE": [12.0, 20.0],
        }
    )

    labeled = add_chaperone_delta_columns(
        df,
        baseline_col="minus",
        tf_col="TF",
        groe_col="GroE",
        kje_col="KJE",
    )

    labeled = add_chaperone_rescue_labels(labeled, threshold=20.0)

    assert labeled.loc[0, "delta_TF"] == 25.0
    assert bool(labeled.loc[0, "rescued_by_TF"]) is True
    assert bool(labeled.loc[0, "rescued_by_any"]) is True
    assert labeled.loc[0, "best_chaperone"] == "TF"

    assert labeled.loc[1, "delta_GroE"] == 35.0
    assert bool(labeled.loc[1, "rescued_by_GroE"]) is True
    assert labeled.loc[1, "best_chaperone"] == "GroE"
