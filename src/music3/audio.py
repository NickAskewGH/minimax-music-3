"""MP3 copies of the lossless WAV master, without resampling."""

from pathlib import Path
import shutil
import subprocess
import tempfile


def export_mp3(source: Path, destination: Path, *, overwrite: bool = False) -> Path:
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise RuntimeError(f"WAV not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("MP3 export requires FFmpeg with libmp3lame. Install FFmpeg; the WAV remains available.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete files, including when several downloads request an export.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".mp3", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source.resolve()), "-map", "0:a:0", "-vn", "-c:a", "libmp3lame",
             "-b:a", "320k", "-f", "mp3", str(temporary.resolve())],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode or not temporary.stat().st_size:
            raise RuntimeError(f"MP3 export failed: {result.stderr.strip() or 'encoder produced no audio'}")
        if overwrite:
            temporary.replace(destination)
        else:
            # Atomic no-clobber publication; preserve an existing user file even in a race.
            destination.hardlink_to(temporary)
        return destination
    except FileExistsError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"MP3 export failed: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def ensure_mp3(source: Path) -> Path:
    destination = source.with_suffix(".mp3")
    if destination.is_file():
        return destination
    try:
        return export_mp3(source, destination)
    except FileExistsError:
        return destination
