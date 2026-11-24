"""Synthetic fixtures for unit tests only; never experimental model results."""
import pytest

from training.contracts import EXPERIMENT_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION
from training.targets import build_targets


@pytest.fixture
def synthetic_observation():
    def make(nfe=4, quality_pass=True, **changes):
        row = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "record_status": "measured", "experiment_id": "unit-test-only",
            "comparison_id": "comparison-test", "prompt_id": "prompt-test",
            "prompt_family_id": "family-test", "speaker_id": "speaker-test",
            "text": "Synthetic fixture 123!", "patch_id": "patch-test", "patch_index": 0,
            "nfe": nfe, "seed": 42, "checkpoint_id": "test-checkpoint",
            "conditioning_id": "test-conditioning", "solver": "test-solver",
            "anchor_state_hash": "test-state", "noise_hash": "test-noise",
            "quality_measurements": {}, "quality_pass": quality_pass,
            "label_protocol": "synthetic-unit-test-v1" if quality_pass is not None else None,
            "audio_valid": True, "not_truncated": True, "counterfactual_reviewed": True,
            "pre_generation_features": {"planned_patch_count": 2},
        }
        row.update(changes)
        return row
    return make


@pytest.fixture
def synthetic_examples(synthetic_observation):
    rows = []
    for family in range(6):
        for variant in range(2):
            pid = f"p{family}-{variant}"
            for patch in range(2):
                for speaker in ("a", "b"):
                    for seed in (11, 22):
                        cid = f"{pid}:{patch}:{speaker}:{seed}"
                        for nfe in (4, 8):
                            rows.append(synthetic_observation(
                                nfe=nfe, quality_pass=(patch == 0 or nfe == 8),
                                comparison_id=cid, prompt_id=pid, prompt_family_id=f"family-{family}",
                                text=f"Fixture {'word ' * (family + 1)}{variant}!",
                                patch_id=f"{pid}:{patch}", patch_index=patch,
                                speaker_id=speaker, seed=seed, conditioning_id=f"cond-{speaker}",
                                anchor_state_hash=f"state-{cid}", noise_hash=f"noise-{seed}",
                            ))
    return build_targets(rows)
