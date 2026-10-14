import hashlib
import json
import shutil
import subprocess

from click.testing import CliRunner
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from music3.audio import export_mp3
from music3.cli import app as cli
from music3.server import create_app


@pytest.fixture
def wav(tmp_path):
    source = tmp_path / "source.wav"
    t = np.arange(44100) / 44100
    sf.write(source, np.column_stack([.2*np.sin(2*np.pi*440*t), .2*np.sin(2*np.pi*880*t)]), 44100)
    return source


def test_mp3_export_cli_and_audio_roundtrip(wav, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the real encoder test")
    original = hashlib.sha256(wav.read_bytes()).digest()
    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(wav)])
    assert result.exit_code == 0, result.output
    mp3 = wav.with_suffix(".mp3")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(mp3)
    ]))["streams"][0]
    assert probe["codec_name"] == "mp3"
    assert probe["sample_rate"] == "44100" and probe["channels"] == 2
    assert int(probe["bit_rate"]) == 320000
    decoded = tmp_path / "decoded.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), str(decoded)], check=True)
    audio, sr = sf.read(decoded)
    reference, _ = sf.read(wav)
    assert sr == 44100 and audio.shape == reference.shape
    assert np.isfinite(audio).all()
    for channel in range(2):
        assert np.corrcoef(audio[:, channel], reference[:, channel])[0, 1] > .99
    assert hashlib.sha256(wav.read_bytes()).digest() == original
    saved = mp3.read_bytes()
    assert runner.invoke(cli, ["export", str(wav)]).exit_code != 0
    assert mp3.read_bytes() == saved
    assert runner.invoke(cli, ["export", str(wav), "--force"]).exit_code == 0


def test_encoder_failure_preserves_existing_files(wav, tmp_path, monkeypatch):
    destination = tmp_path / "song.mp3"
    destination.write_bytes(b"existing export")
    monkeypatch.setattr("music3.audio.subprocess.run", lambda *a, **k:
                        subprocess.CompletedProcess([], 1, stderr="encoder failed"))
    monkeypatch.setattr("music3.audio.shutil.which", lambda name: "/test/ffmpeg")
    with pytest.raises(RuntimeError, match="encoder failed"):
        export_mp3(wav, destination, overwrite=True)
    assert destination.read_bytes() == b"existing export"
    assert set(tmp_path.iterdir()) == {wav, destination}
    monkeypatch.setattr("music3.audio.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires FFmpeg"):
        export_mp3(wav, tmp_path / "missing.mp3")


def test_existing_job_mp3_download(wav, tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    job_id = "a" * 32
    folder = tmp_path / "outputs" / job_id
    folder.mkdir(parents=True)
    shutil.copyfile(wav, folder / "song.wav")
    status = folder / "status.json"
    status.write_text(json.dumps({"id": job_id, "stage": "complete"}))
    with TestClient(create_app(tmp_path / "model", tmp_path / "outputs")) as client:
        response = client.get(f"/api/jobs/{job_id}/files/song.mp3")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == (folder / "song.mp3").read_bytes()
        assert response.content
        assert client.get(f"/api/jobs/{job_id}/files/song.mp3").content == response.content
        status.write_text(json.dumps({"id": job_id, "stage": "failed"}))
        assert client.get(f"/api/jobs/{job_id}/files/song.mp3").status_code == 409
