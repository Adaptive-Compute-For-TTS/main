import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import compute_wer, normalize, wer_cer  # noqa: E402
from wer_scoring import official_columns, score_clip  # noqa: E402

INTRO = "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time."
KETTLE = "The kettle is already on the stove."


def test_identical_text_is_zero():
    s = score_clip(KETTLE, KETTLE)
    assert s["wer_full"] == s["wer_no_intro"] == 0.0
    assert s["cer_full"] == 0.0


def test_one_wrong_word_in_seven():
    s = score_clip(KETTLE, "The kettle is already on the stone.")
    assert s["wer_full"] == pytest.approx(1 / 7)
    assert s["subs_full"] == 1


def test_empty_transcript_is_one():
    s = score_clip(KETTLE, "")
    for v in ("full", "no_intro"):
        assert s[f"wer_{v}"] == 1.0 and s[f"cer_{v}"] == 1.0
        assert s[f"dels_{v}"] == s[f"trailing_dels_{v}"] == s["ref_words"] == 7
    assert s["intro_text"] == ""


@pytest.mark.parametrize("written, spoken", [
    ("$42.75", "forty two dollars and seventy five cents"),
    ("6:15 a.m.", "6:15 AM"),
    ("St. Clair Ave.", "Saint Clair Avenue"),
])
def test_normalization_pairs(written, spoken):
    w = compute_wer(written, spoken)
    assert w["wer"] == 0.0, f"ref_norm={w['ref_norm']!r} hyp_norm={w['hyp_norm']!r}"


def test_no_intro_drops_leading_phrase():
    s = score_clip(KETTLE, f"{INTRO} {KETTLE}")
    assert s["wer_full"] > 1.0
    assert s["wer_no_intro"] == 0.0
    assert s["intro_text"] == normalize(INTRO)


def test_no_intro_never_drops_matching_words():
    # Nothing to drop: the transcript starts with the reference.
    assert score_clip(KETTLE, KETTLE)["intro_text"] == ""
    # Only intro words are dropped: "the kettle" stays, and leading words that are not the
    # intro ("hello there") are scored as insertions, not set aside.
    s = score_clip(KETTLE, "hey adam here the kettle is already on the stone")
    assert s["intro_text"] == "hey adam here"
    assert s["subs_no_intro"] == 1 and s["wer_no_intro"] == pytest.approx(1 / 7)
    s = score_clip(KETTLE, "hello there the kettle is already on the stone")
    assert s["intro_text"] == "" and s["ins_no_intro"] == 2


def test_no_intro_only_drops_leading_words():
    # Inserted words in the middle are not removed.
    s = score_clip(KETTLE, "The kettle is very already on the stove.")
    assert s["intro_text"] == ""
    assert s["ins_no_intro"] == 1


def test_trailing_dels_counts_only_end_deletions():
    s = score_clip(KETTLE, "The kettle is")
    assert s["trailing_dels_full"] == 4
    s = score_clip(KETTLE, "The kettle on the stove.")  # deletions in the middle only
    assert s["dels_full"] == 2 and s["trailing_dels_full"] == 0
    s = score_clip(KETTLE, "The kettle on the")  # middle and end
    assert s["dels_full"] == 3 and s["trailing_dels_full"] == 1


def test_truncated_after_intro():
    s = score_clip(KETTLE, f"{INTRO} The kettle is")
    assert s["wer_no_intro"] == pytest.approx(4 / 7)
    assert s["trailing_dels_no_intro"] == 4


def test_capping_never_changes_uncapped_columns():
    s = score_clip(KETTLE, f"{INTRO} {KETTLE}")
    before = dict(s)
    o = official_columns(s, "full")
    assert s == before
    assert o["wer"] == 1.0 and o["wer_uncapped"] == round(s["wer_full"], 4) > 1.0
    assert o["cer"] == 1.0 and o["cer_uncapped"] == round(s["cer_full"], 4) > 1.0
    o = official_columns(s, "no_intro")
    assert o["wer"] == o["wer_uncapped"] == 0.0 and o["wer_version"] == "no_intro"


# --- intro alignment (concatenated reference) and missing_end_words ---

def test_intro_plus_perfect_sentence():
    s = score_clip(KETTLE, f"{INTRO} {KETTLE}")
    assert s["wer_no_intro"] == 0.0 and s["missing_end_words_no_intro"] == 0
    assert s["intro_text"] == normalize(INTRO)


def test_intro_only_sentence_missing():
    s = score_clip(KETTLE, INTRO)
    assert s["wer_no_intro"] == 1.0
    assert s["missing_end_words_no_intro"] == s["ref_words"] == 7
    assert s["missing_end_words_full"] == 7
    assert s["intro_text"] == normalize(INTRO)


def test_partial_intro_plus_sentence():
    s = score_clip(KETTLE, f"Hey Adam here, let's create something. {KETTLE}")
    assert s["intro_text"] == "hey adam here let us create something"
    assert s["wer_no_intro"] == 0.0


def test_no_intro_at_all_equals_full():
    for hyp in (KETTLE, "The kettle is already on the stone.", "The kettle is"):
        s = score_clip(KETTLE, hyp)
        assert s["intro_text"] == ""
        for m in ("wer", "cer", "subs", "dels", "ins", "trailing_dels", "missing_end_words"):
            assert s[f"{m}_no_intro"] == s[f"{m}_full"], m


def test_cut_off_ending_missing_end_words():
    s = score_clip(KETTLE, f"{INTRO} The kettle is already on the st")
    assert s["missing_end_words_no_intro"] == 1  # "stove" never matched
    assert s["trailing_dels_no_intro"] == 0      # jiwer calls it a substitution
    s = score_clip(KETTLE, f"{INTRO} The kettle is")
    assert s["missing_end_words_no_intro"] == 4


def test_long_reference_cut_inside_intro():
    ref = ("When the storm finally cleared, the whole street came outside at once. Children splashed "
           "through the puddles, someone carried a ladder over to the fallen branch, and an older man "
           "stood on his porch counting the shingles that had blown off his roof.")
    s = score_clip(ref, "Hey, Adam here. Let's create something that feels real, sounds human.")
    assert s["missing_end_words_no_intro"] == s["ref_words"] == 43
    assert s["wer_no_intro"] == 1.0
    # In the full version the intro word "that" happens to hit "...shingles that had blown off his roof".
    assert s["missing_end_words_full"] == 5


PROMPTS = json.loads((Path(__file__).resolve().parents[1] / "prompts/benchmark_prompts_v1.json").read_text())


@pytest.mark.parametrize("sentence", [KETTLE] + [p["text"] for p in PROMPTS])
def test_correct_sentence_never_attributed_to_intro(sentence):
    s = score_clip(sentence, sentence)
    assert s["intro_text"] == ""
    assert s["wer_no_intro"] == s["wer_full"] == 0.0


# --- format rules added 2026-10-14 (written/spoken pairs from the full run) ---

@pytest.mark.parametrize("written, spoken", [
    ("St. Clair Ave. is closed from Jan. to Mar., so take the TTC",
     "St. Clair Ave is closed from Jan to Mar, so take the TTC"),
    ("leaning against the doorframe.", "leaning against the door frame."),
    ("a quetzal feather headdress", "a Quetzalfeather headdress,"),
    ("a quetzal feather headdress", "a quetzal-feather headdress"),
    ("Our startup's MVP uses a REST API", "Our start-ups MVP uses a REST API"),
    ("Our startup's MVP uses a REST API", "Our startups MVP uses a REST API"),
    ("Our startup's MVP uses a REST API", "Our start-up's MVP uses a REST API"),
])
def test_new_format_rules(written, spoken):
    w = compute_wer(written, spoken)
    assert w["wer"] == 0.0, f"ref_norm={w['ref_norm']!r} hyp_norm={w['hyp_norm']!r}"


@pytest.mark.parametrize("written, spoken", [
    ("Call me back at 555-0147.", "Call me back at 5550147."),
    ("Call me back at 555-0147.", "Call me back at 555 0147."),
    ("Call me back at 555-0147.", "Call me back at five five five zero one four seven."),
    ("lunch at 12:05", "lunch at 12 05"),
])
def test_leading_zero_digit_groups(written, spoken):
    w = compute_wer(written, spoken)
    assert w["wer"] == 0.0, f"ref_norm={w['ref_norm']!r} hyp_norm={w['hyp_norm']!r}"


def test_digit_groups_never_join_into_a_year():
    assert normalize("On March 3, 2024, it snowed") == "on march 3 2024 it snowed"
    assert normalize("between 2023 2024") == "between 2023 2024"
    assert normalize("6 15") == "615"   # times still join
    # "3rd" vs "3" is one substitution, not two from a merged "32024" token
    w = compute_wer("On March 3rd, 2024, it snowed", "On March 3, 2024, it snowed")
    assert w["subs"] + w["dels"] + w["ins"] == 1


def test_month_rules_leave_full_month_names_alone():
    assert normalize("in March and January") == "in march and january"


# --- strict intro attribution (only through the last exact intro match) ---

def test_misheard_first_word_not_attributed_to_intro():
    s = score_clip("Siobhan met Joaquín in Reykjavík.", "Savan met Joaquin in Reykjavík.")
    assert s["intro_text"] == ""
    assert s["subs_no_intro"] == s["subs_full"] == 1
    s = score_clip("Pass the quinoa and the gnocchi.", "past the quinoa and the gnocchi.")
    assert s["intro_text"] == ""


def test_misheard_first_word_after_intro_stays_in_scored_part():
    s = score_clip("Siobhan met Joaquín in Reykjavík.", f"{INTRO} see you bon met Joaquin in Reykjavík.")
    assert s["intro_text"] == normalize(INTRO)
    assert s["wer_no_intro"] > 0


# --- heard-portion WER ---

def test_heard_ignores_cut_off_ending():
    s = score_clip(KETTLE, f"{INTRO} The kettle is")
    assert s["heard_ref_words_no_intro"] == 3 and s["heard_errors_no_intro"] == 0
    assert s["wer_heard_no_intro"] == 0.0
    s = score_clip(KETTLE, "The kettle is all ready on the st")  # error inside the heard part, cut ending
    assert s["heard_ref_words_full"] == 6 and s["missing_end_words_full"] == 1
    assert s["heard_errors_full"] == 2  # already -> all ready


def test_heard_nothing_heard_is_nan():
    s = score_clip(KETTLE, "")
    assert s["heard_ref_words_no_intro"] == 0 and s["wer_heard_no_intro"] != s["wer_heard_no_intro"]
    s = score_clip(KETTLE, INTRO)
    assert s["heard_ref_words_no_intro"] == 0


def test_heard_equals_wer_when_nothing_is_cut():
    s = score_clip(KETTLE, "The kettle is already on the stone stove.")
    assert s["missing_end_words_full"] == 0
    assert s["wer_heard_full"] == s["wer_full"]


# --- corpus CER ---

def test_cer_counts_match_jiwer_and_corpus_sums():
    import jiwer
    from wer_scoring import corpus_cer
    a = score_clip(KETTLE, "The kettle is already on the stone.")
    b = score_clip(KETTLE, "")
    ref = normalize(KETTLE)
    assert a["cer_full"] == pytest.approx(jiwer.cer(ref, normalize("The kettle is already on the stone.")))
    assert a["ref_chars_full"] == len(ref) and b["char_errors_full"] == b["ref_chars_full"] == len(ref)
    c = corpus_cer([a, b], "full")
    assert c["cer"] == pytest.approx((a["char_errors_full"] + len(ref)) / (2 * len(ref)))


# --- wer_cer() ---

@pytest.mark.parametrize("transcript", [
    KETTLE, "The kettle is already on the stone.", "the kettle the kettle is on the stove", "",
])
def test_wer_cer_matches_score_clip_full(transcript):
    s = score_clip(KETTLE, transcript)
    assert wer_cer(KETTLE, transcript) == pytest.approx((s["wer_full"], s["cer_full"]))


def test_wer_cer_empty_reference_is_nan():
    assert all(math.isnan(x) for x in wer_cer("", "something"))
    s = score_clip("", "something")
    assert math.isnan(s["wer_full"]) and math.isnan(s["cer_full"])
