import json
from pathlib import Path
import subprocess
import shutil
import sys
import time

from click.testing import CliRunner
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from music3.cli import app as cli
from music3.engine import inspect_model
from music3.jobs import JobManager, TERMINAL
from music3.schema import PRESETS, SongRequest
from music3.server import create_app


@pytest.fixture
def song():
    return SongRequest(**PRESETS["daybreak"], randomize_seed=False, seed=7)


def test_prompt_keeps_instructions_out_of_lyrics(song):
    assert "Global Metadata" in song.caption()
    assert "BPM: 96" in song.caption()
    assert "[verse] Solo guitar" in song.caption()
    assert "Solo guitar" not in song.rendered_lyrics()
    assert song.rendered_lyrics().startswith("[verse]\nMorning")
    assert SongRequest.model_validate_json(song.model_dump_json()) == song


@pytest.mark.parametrize("patch", [
    {"duration": 301}, {"duration": 4}, {"steps": 3}, {"steps": 61},
    {"guidance": 0.9}, {"guidance": 4.1}, {"headroom": -1}, {"headroom": 61},
    {"seed": -1}, {"seed": 2**31}, {"bpm": 0}, {"duration": float("nan")},
    {"guidance": float("inf")}, {"offload": "invalid"}, {"unrecognized": 1},
    {"sections": [], "lyrics": "[verse] These words would be lost"},
    {"lyrics": "Duplicate raw lyrics alongside the chart"},
    {"sections": [{"name": "verse]\nInjected", "lyrics": "test"}]},
])
def test_invalid_requests_rejected(song, patch):
    with pytest.raises(ValidationError):
        SongRequest.model_validate({**song.model_dump(), **patch})


def test_full_duration_and_studio_limits(song):
    request = SongRequest.model_validate({**song.model_dump(), "duration": 300, "steps": 60,
        "guidance": 4, "headroom": 60, "seed": 2**31 - 1})
    assert request.duration == 300


def test_instrumental_preserves_directions(song):
    data = song.model_dump()
    data["instrumental"] = True
    instrumental = SongRequest(**data)
    assert instrumental.rendered_lyrics() == "[instrumental]"
    assert "no vocals" in instrumental.caption()
    assert "[chorus] Brushed drums" in instrumental.caption()


def test_seed_resolution(monkeypatch, song):
    monkeypatch.setattr("music3.schema.secrets.randbelow", lambda n: 123)
    assert song.resolved().seed == 7
    random_song = song.model_copy(update={"randomize_seed": True}).resolved()
    assert random_song.seed == 123
    assert random_song.randomize_seed is False


def test_cli_has_every_setting_and_matching_preview(song, tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(cli, ["generate", "--help"])
    assert help_result.exit_code == 0
    for name in SongRequest.model_fields:
        flag = "--sections-file" if name == "sections" else "--" + name.replace("_", "-")
        assert flag in help_result.output
    config = tmp_path / "song.json"
    config.write_text(song.model_dump_json())
    result = runner.invoke(cli, ["generate", "--config", str(config), "--duration", "300",
        "--steps", "60", "--guidance", "4", "--headroom", "60", "--fixed-seed", "--seed", "42", "--dry-run"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["request"]["duration"] == 300
    assert data["request"]["seed"] == 42
    assert data["request"]["randomize_seed"] is False
    assert data["caption"] == song.caption()
    result = runner.invoke(cli, ["preview", str(config)])
    assert json.loads(result.output)["lyrics"] == song.rendered_lyrics()


def test_cli_files_and_bad_settings(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[chorus]\nWords worth singing")
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--preset", "daybreak", "--lyrics-file", str(lyrics), "--dry-run"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["request"]["sections"] == []
    assert data["lyrics"] == lyrics.read_text()
    assert runner.invoke(cli, ["generate", "--duration", "301", "--dry-run"]).exit_code != 0


def test_missing_and_truncated_weights(tmp_path):
    report = inspect_model(tmp_path)
    assert not report["model_ready"]
    assert "Missing modular_model_index.json" in report["issues"]
    folder = tmp_path / "language_model"
    folder.mkdir()
    (folder / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1")
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"weight":"missing.safetensors"}}))
    report = inspect_model(tmp_path)
    assert any("Missing language_model/missing.safetensors" in issue for issue in report["issues"])
    assert any("not a safetensors" in issue for issue in report["issues"])


@pytest.fixture
def fake_worker(monkeypatch):
    """Exercise actual subprocess/queue handling without consuming GPU time."""
    original = subprocess.Popen
    worker = '''
import json, sys, time, pathlib, wave
folder = pathlib.Path(sys.argv[1])
song = json.loads((folder / 'request.json').read_text())
print(json.dumps({'stage':'loading','message':'Loading test worker'}), flush=True)
if song['title'] == 'slow': time.sleep(30)
if song['title'] == 'fail':
    print(json.dumps({'stage':'failed','message':'Expected worker failure'}), flush=True)
    sys.exit(1)
with wave.open(str(folder/'song.wav'), 'wb') as wav:
    wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(44100); wav.writeframes(b'\\0'*400)
print(json.dumps({'stage':'complete','message':'Done','progress':1,'audio':'song.wav'}), flush=True)
'''
    def launch(args, **kwargs):
        if len(args) > 2 and args[1:3] == ["-m", "music3.worker"]:
            args = [sys.executable, "-u", "-c", worker, args[-2]]
        return original(args, **kwargs)
    monkeypatch.setattr("music3.jobs.subprocess.Popen", launch)


def wait_job(manager, job_id, predicate=lambda j: j["stage"] in TERMINAL):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if predicate(job):
            return job
        time.sleep(0.02)
    raise AssertionError(f"Job did not reach expected state: {manager.get(job_id)}")


def test_queue_cancel_failure_and_persistence(tmp_path, song, fake_worker):
    manager = JobManager(tmp_path / "model", tmp_path / "outputs")
    try:
        slow = manager.submit(song.model_copy(update={"title":"slow"}))
        wait_job(manager, slow["id"], lambda j: j["stage"] == "loading")
        queued = manager.submit(song)
        assert manager.get(queued["id"])["stage"] == "queued"
        assert manager.cancel(queued["id"])["stage"] == "cancelled"
        assert manager.cancel(slow["id"])["stage"] == "cancelled"
        failed = manager.submit(song.model_copy(update={"title":"fail"}))
        assert wait_job(manager, failed["id"])["stage"] == "failed"
        success = manager.submit(song)
        assert wait_job(manager, success["id"])["stage"] == "complete"
        assert (manager.directory(success["id"]) / "song.wav").is_file()
        saved = SongRequest.model_validate_json((manager.directory(success["id"]) / "request.json").read_text())
        assert saved.seed == 7 and not saved.randomize_seed
        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--output-dir", str(manager.output_dir)])
        assert result.exit_code == 0
        assert success["id"] in {item["id"] for item in json.loads(result.output)}
        result = runner.invoke(cli, ["status", success["id"], "--output-dir", str(manager.output_dir)])
        assert json.loads(result.output)["stage"] == "complete"
    finally:
        manager.close()
    restarted = JobManager(tmp_path / "model", tmp_path / "outputs")
    try:
        assert restarted.get(success["id"])["stage"] == "complete"
        assert len(restarted.list()) == 4
    finally:
        restarted.close()


def test_other_process_and_stale_owner(tmp_path, song, fake_worker):
    manager = JobManager(tmp_path, tmp_path / "output")
    other = JobManager(tmp_path, tmp_path / "output")
    try:
        job = manager.submit(song.model_copy(update={"title":"slow"}))
        wait_job(manager, job["id"], lambda j: j["stage"] == "loading")
        assert other.get(job["id"])["stage"] == "loading"
        with pytest.raises(ValueError, match="another running"):
            other.cancel(job["id"])
        manager.cancel(job["id"])
        status = manager.directory(job["id"]) / "status.json"
        data = json.loads(status.read_text())
        data.update(stage="loading", owner_pid=999999999)
        status.write_text(json.dumps(data))
        assert other.get(job["id"])["stage"] == "interrupted"
    finally:
        manager.close()
        other.close()


def test_api_roundtrip_downloads_and_validation(tmp_path, song, fake_worker):
    app = create_app(tmp_path / "model", tmp_path / "output")
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert set(client.get("/api/schema").json()["properties"]) == set(SongRequest.model_fields)
        preview = client.post("/api/preview", json=song.model_dump())
        assert preview.json() == {"caption":song.caption(), "lyrics":song.rendered_lyrics()}
        assert client.post("/api/jobs", json={**song.model_dump(), "steps":61}).status_code == 422
        assert client.post("/api/jobs", json=song.model_dump(), headers={"Origin":"http://evil.invalid"}).status_code == 403
        assert client.get("/", headers={"Host":"evil.invalid"}).status_code == 400
        result = client.post("/api/jobs", json=song.model_dump())
        assert result.status_code == 202
        job_id = result.json()["id"]
        assert wait_job(app.state.manager, job_id)["stage"] == "complete"
        assert client.get(f"/api/jobs/{job_id}/files/song.wav").content.startswith(b"RIFF")
        settings = client.get(f"/api/jobs/{job_id}/files/request.json").json()
        assert SongRequest(**settings).caption() == song.caption()
        assert client.get(f"/api/jobs/{job_id}/files/secrets.env").status_code == 404
        assert client.get("/api/jobs/not-a-job").status_code == 404
        assert client.post(f"/api/jobs/{job_id}/cancel").json()["stage"] == "complete"


def test_cli_mp3_output_preserves_wav_and_settings(tmp_path, fake_worker):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    destination = tmp_path / "export.mp3"
    outputs = tmp_path / "outputs"
    result = CliRunner().invoke(cli, ["generate", "--preset", "daybreak", "--fixed-seed", "--seed", "7",
        "--output-dir", str(outputs), "--output", str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.stat().st_size > 0
    assert json.loads(destination.with_suffix(".json").read_text())["seed"] == 7
    job = next(outputs.iterdir())
    assert (job / "song.wav").is_file()
    assert (job / "song.mp3").read_bytes() == destination.read_bytes()
