"""CPU-safe integration fixtures; these are not speech-quality/GPU equivalence tests."""
from pathlib import Path
from types import SimpleNamespace
import wave

from fastapi.testclient import TestClient

from backend.app import repository
from backend.app.database import SessionLocal
from backend.app.main import create_app
from backend.app.workers.local_worker import LocalWorker
from backend.app.routes import runs as routes
from backend.app.schemas import RunStatus


def test_queue_consumes_generates_persists_and_serves_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "AUDIO_DIR", tmp_path)
    calls = []
    def processor(run):
        calls.append((run.id, run.prompt, run.nfe))
        path = tmp_path / f"{run.id}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)
            audio.writeframes(b'\0\0' * 24000)
        return SimpleNamespace(status="completed", result={"nfe":run.nfe, "test_fixture":True, "validation":{"ok":True,"flags":[]}, "telemetry":{"fingerprint":{"device":"cpu"}}, "quality":{"scored":True,"wer":0.25,"cer":0.1}}, audio_path=str(path), error_message=None)
    worker = LocalWorker(mode="gpu", processor=processor)
    with TestClient(create_app(worker)) as client:
        submitted = client.post('/runs', json={"prompt":"A processor integration check.","nfe":32})
        assert submitted.status_code == 202
        id = submitted.json()['id']
        worker.wait_until_idle()
        data = client.get(f'/runs/{id}').json()
        assert data['status'] == 'completed'
        assert data['audio_url'] == f'/runs/{id}/audio'
        assert data['result']['quality']['wer'] == 0.25
        assert data['result']['telemetry']['fingerprint']['device'] == 'cpu'
        assert data['result']['validation']['ok'] is True
        assert calls == [(id, 'A processor integration check.', 32)]
        with SessionLocal() as database:
            row = repository.get_run(database, id)
            assert row.status == RunStatus.COMPLETED and row.audio_path == str(tmp_path / f'{id}.wav')
        audio = client.get(data['audio_url'])
        assert audio.status_code == 200 and audio.content[:4] == b'RIFF'
        ranged = client.get(data['audio_url'], headers={'Range':'bytes=0-43'})
        assert ranged.status_code == 206 and len(ranged.content) == 44


def test_rejected_wav_keeps_validation_and_telemetry_in_sqlite():
    result = {"validation":{"ok":False,"flags":["silent"]}, "telemetry":{"scope":"generation_only"}, "quality":{"status":"skipped"}}
    worker = LocalWorker(mode="gpu", processor=lambda run: SimpleNamespace(status="failed", result=result, audio_path=None, error_message="WAV failed validation: silent"))
    with TestClient(create_app(worker)) as client:
        id = client.post('/runs', json={"prompt":"Rejected audio test.", "nfe":4}).json()['id']
        worker.wait_until_idle()
        data = client.get(f'/runs/{id}').json()
        assert data['status'] == 'failed' and data['audio_url'] is None
        assert data['result'] == result
        with SessionLocal() as database:
            assert repository.get_run(database,id).result == result


def test_generation_failure_is_saved_and_next_job_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "AUDIO_DIR", tmp_path)
    def processor(run):
        raise RuntimeError('Model could not load')
    worker = LocalWorker(mode='gpu', processor=processor)
    with TestClient(create_app(worker)) as client:
        for prompt in ('First request', 'Second request'):
            id = client.post('/runs', json={'prompt':prompt, 'nfe':4}).json()['id']
            worker.wait_until_idle()
            data = client.get(f'/runs/{id}').json()
            assert data['status'] == 'failed' and 'Model could not load' in data['error_message']
            assert data['audio_url'] is None
            assert client.get(f'/runs/{id}/audio').status_code == 404


def test_audio_cannot_serve_arbitrary_files(tmp_path, monkeypatch, client):
    safe = tmp_path / 'allowed'
    safe.mkdir()
    monkeypatch.setattr(routes, 'AUDIO_DIR', safe)
    outside = tmp_path / 'private.wav'; outside.write_bytes(b'secret')
    id = client.post('/runs', json={'prompt':'Check path safety','nfe':20}).json()['id']
    with SessionLocal() as database:
        repository.mark_run_running(database, id)
        repository.mark_run_completed(database, id, {}, str(outside))
    assert client.get(f'/runs/{id}/audio').status_code == 404
    assert client.get('/runs/00000000-0000-0000-0000-000000000000/audio').status_code == 404
