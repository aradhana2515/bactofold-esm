from bactofold.features import sequence_features


def test_sequence_features_basic():
    feats = sequence_features("MKTAYIAK")
    assert feats["length"] == 8
    assert "mean_hydropathy" in feats
    assert feats["aa_K"] > 0
