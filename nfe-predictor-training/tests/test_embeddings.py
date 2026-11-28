"""Static embeddings on a tiny synthetic model; optional parity with the real libraries."""
import json
import os
import struct

import numpy as np
import pytest

from experiments.feature_ablation import run_ablation
from experiments.ablation_data import from_patch_targets
from experiments.feature_sets import build, with_static_embeddings
from features.embeddings import StaticEmbeddings, WordPiece, bert_normalize, bert_pre_tokenize, read_safetensors
from training.train import TrainingConfig

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "un", "##believ", "##able", ",", "!", "cafe"]


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


@pytest.fixture
def tiny_model_dir(tmp_path):
    table = np.arange(len(VOCAB) * 3, dtype=np.float32).reshape(len(VOCAB), 3)
    _write_safetensors(tmp_path / "model.safetensors", {"embeddings": table})
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "normalizer": {"type": "BertNormalizer", "lowercase": True, "strip_accents": None},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100, "vocab": {t: i for i, t in enumerate(VOCAB)}},
    }))
    (tmp_path / "config.json").write_text(json.dumps({"normalize": False}))
    return tmp_path


def test_safetensors_round_trip(tmp_path):
    a = np.random.default_rng(0).normal(size=(4, 2)).astype(np.float32)
    _write_safetensors(tmp_path / "m.safetensors", {"embeddings": a, "other": a[:1]})
    loaded = read_safetensors(tmp_path / "m.safetensors")
    assert np.array_equal(loaded["embeddings"], a.astype(np.float64)) and loaded["other"].shape == (1, 2)


def test_normalize_and_pre_tokenize():
    assert bert_normalize("Café\tNAÏVE\x07") == "cafe naive"
    assert bert_normalize("東京") == " 東  京 "
    assert bert_pre_tokenize("hello, world!  it's") == ["hello", ",", "world", "!", "it", "'", "s"]


def test_wordpiece(tiny_model_dir):
    wp = WordPiece.from_tokenizer_json(tiny_model_dir / "tokenizer.json")
    ids = {t: i for i, t in enumerate(VOCAB)}
    assert wp.encode("Hello, Unbelievable world!") == [
        ids["hello"], ids[","], ids["un"], ids["##believ"], ids["##able"], ids["world"], ids["!"]]
    assert wp.encode("Café xyz") == [ids["cafe"], ids["[UNK]"]]
    assert wp.encode("h" * 101) == [ids["[UNK]"]]


def test_rejects_non_wordpiece(tiny_model_dir):
    spec = json.loads((tiny_model_dir / "tokenizer.json").read_text())
    (tiny_model_dir / "tokenizer.json").write_text(json.dumps({**spec, "model": {**spec["model"], "type": "BPE"}}))
    with pytest.raises(ValueError, match="WordPiece"):
        StaticEmbeddings.load(tiny_model_dir)


def test_embed_is_mean_of_known_tokens(tiny_model_dir):
    model = StaticEmbeddings.load(tiny_model_dir, name="tiny")
    table = model.table
    assert np.allclose(model.embed("hello world xyz"), table[[4, 5]].mean(axis=0))  # [UNK] dropped
    assert np.allclose(model.embed("hello!!"), table[[4, 10, 10]].mean(axis=0))  # punctuation counts
    assert np.array_equal(model.embed(""), np.zeros(3))
    normalized = StaticEmbeddings(model.name, model.tokenizer, table, normalize=True)
    assert np.isclose(np.linalg.norm(normalized.embed("hello")), 1.0)


def test_embeddings_compose_in_specs(tiny_model_dir, synthetic_examples):
    fitters = with_static_embeddings({"tiny": StaticEmbeddings.load(tiny_model_dir, name="tiny")})
    texts = ["hello world", "hello, unbelievable!", "cafe world"]
    fitted = build("text5+pca2(tiny)", fitters)(texts)
    assert len(fitted.names) == 7
    assert build("tiny", fitters)(texts).shipped_params == len(VOCAB) * 3
    with pytest.raises(ValueError, match="clash"):
        with_static_embeddings({"text5": StaticEmbeddings.load(tiny_model_dir)})
    report = run_ablation(from_patch_targets(synthetic_examples), ["tiny", "pca2(tiny)"], seeds=(0,),
                          config=TrainingConfig(epochs=1), fixed_nfe=8, fallback_nfe=8, base_fitters=fitters)
    assert [fs["dims"] for fs in report["feature_sets"]] == [3, 2]


def test_pca_folds_into_a_smaller_identical_table(tiny_model_dir):
    model = StaticEmbeddings.load(tiny_model_dir, name="tiny")
    rng = np.random.default_rng(0)
    model = StaticEmbeddings(model.name, model.tokenizer, rng.normal(size=model.table.shape), normalize=True)
    fitters = with_static_embeddings({"tiny": model})
    train = ["hello world", "hello, unbelievable!", "cafe world!", "un world", "hello cafe"]
    texts = [*train, "", "xyz", "world, world, hello"]

    folded = build("pca2(tiny.raw)", fitters)(train)
    assert folded.token_mean is not None and folded.shipped_params == len(VOCAB) * 2 + 2
    # Same numbers as standardizing and projecting the raw mean afterwards.
    raw = StaticEmbeddings(model.name, model.tokenizer, model.table, normalize=False)
    x = np.stack([raw.embed(t) for t in train])
    mean, std = x.mean(axis=0), np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    _, _, vt = np.linalg.svd((x - mean) / std, full_matrices=False)
    signs = np.where(vt[np.arange(2), np.abs(vt[:2]).argmax(axis=1)] < 0, -1.0, 1.0)
    expected = ((np.stack([raw.embed(t) for t in texts]) - mean) / std) @ (vt[:2] * signs[:, None]).T
    assert np.allclose(folded.matrix(texts), expected)

    # L2-normalized output is not affine in the table, so it keeps the full table.
    unfolded = build("pca2(tiny)", fitters)(train)
    assert unfolded.token_mean is None
    assert unfolded.shipped_params == len(VOCAB) * 3 + 2 * 3 + 3 + 3


REAL_MODEL = os.environ.get("NFE_STATIC_EMBEDDINGS_DIR")


@pytest.mark.skipif(not REAL_MODEL, reason="set NFE_STATIC_EMBEDDINGS_DIR to a model2vec directory")
def test_parity_with_tokenizers_library():
    tokenizers = pytest.importorskip("tokenizers")
    model = StaticEmbeddings.load(REAL_MODEL)
    reference = tokenizers.Tokenizer.from_file(f"{REAL_MODEL}/tokenizer.json")
    for text in ["Call NASA at 555-0142, Friday!", "Café naïve — “quotes” 東京 €5,000!!", "", "a" * 150]:
        assert model.tokenizer.encode(text) == reference.encode(text, add_special_tokens=False).ids
