import subprocess
import sys

import wav_validation
from synth import SR, clean


def test_public_api_is_importable_from_the_package():
    r = wav_validation.validate_audio(clean(), SR)
    assert r["ok"] and r["version"] == wav_validation.__version__
    assert callable(wav_validation.validate_wav)
    assert callable(wav_validation.validate_and_store)
    assert "silent" in wav_validation.FLAGS


def test_fresh_interpreter_import_has_no_cycle():
    code = "import wav_validation, numpy as np; print(wav_validation.validate_audio(np.zeros(48000), 48000)['flags'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "['silent']"


def test_does_not_import_torch():
    code = "import sys, wav_validation; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
