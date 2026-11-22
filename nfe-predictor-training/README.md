# NFE Predictor Training

Training, evaluation, and experimentation code for the adaptive NFE predictor.

## Basic feature extraction

The feature extractor contains inexpensive, deterministic properties that are
available before speech generation:

| Feature | Definition |
| --- | --- |
| `character_count` | All characters, including whitespace |
| `word_count` | Number of whitespace-separated tokens |
| `punctuation_count` | ASCII punctuation characters from `string.punctuation` |
| `digit_count` | Individual numeric characters |
| `average_word_length` | Average token length, including punctuation |

Empty and whitespace-only prompts have an average word length of `0.0`. The
canonical feature ordering is defined by `features.FEATURE_NAMES` and must be
used by both training and inference.

```python
from features import extract_features, features_to_vector

features = extract_features("Hello, world!")
model_input = features_to_vector(features)
```

The optional `context` argument is reserved for future speaker metadata or
embedding features. Basic extraction does not currently use it.

## Tests

Run the test suite from this directory:

```bash
python -m pytest tests -v
```
