"""text5-plus-patch-index-v1 gives exactly what the training code gives.

Expected values were produced by running feature/nfe-predictor-training @ 89c5084
(features/extractor.py: features_to_vector(extract_features(text))). If training changes its features,
regenerate these and bump the schema name.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfe_predictor.features import ContextError, get_extractor

FROM_TRAINING = [
    ("Hello, world!", [13.0, 2.0, 2.0, 0.0, 6.0]),
    ("Hi there.", [9.0, 2.0, 1.0, 0.0, 4.0]),
    ("Call me at 555-0142 on Friday.", [30.0, 6.0, 2.0, 7.0, 4.166666666666667]),
    ("  spaced   out  ", [16.0, 2.0, 0.0, 0.0, 4.5]),
    ("", [0.0, 0.0, 0.0, 0.0, 0.0]),
    ("Ünïcode ½ ٣ digits!?", [20.0, 4.0, 2.0, 1.0, 4.25]),
    ("The scheduler decides how many denoising steps each patch of audio receives today.",
     [82.0, 13.0, 1.0, 0.0, 5.384615384615385]),
]

ex = get_extractor("text5-plus-patch-index-v1")


def test_names_and_order_match_training():
    assert ex.feature_names == ("character_count", "word_count", "punctuation_count", "digit_count",
                                "average_word_length", "patch_index")


@pytest.mark.parametrize("text,expected", FROM_TRAINING)
@pytest.mark.parametrize("patch_index", [0, 7])
def test_values_match_training(text, expected, patch_index):
    got = ex(text, {"patch_index": patch_index})
    assert np.array_equal(got, np.array(expected + [float(patch_index)]))


@pytest.mark.parametrize("context", [{}, {"patch_index": -1}, {"patch_index": 1.0}, {"patch_index": True},
                                     {"patch_index": "3"}, {"patch_index": None}])
def test_patch_index_is_required(context):
    with pytest.raises(ContextError):
        ex.check_context(context)
