"""One decision from the command line:

    python -m nfe_predictor BUNDLE_DIR "text" [--patch-index 0] [--fallback-nfe 32]
"""

import argparse
import json

from nfe_predictor import load_predictor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("text")
    p.add_argument("--patch-index", type=int, default=None, help="for patch-level bundles")
    p.add_argument("--fallback-nfe", type=int, default=32)
    a = p.parse_args()
    predictor = load_predictor(a.bundle, fallback_nfe=a.fallback_nfe)
    print(json.dumps(predictor.predict_budget(
        {"text": a.text} if a.patch_index is None else {"text": a.text, "patch_index": a.patch_index}).as_dict(), indent=2))


if __name__ == "__main__":
    main()
