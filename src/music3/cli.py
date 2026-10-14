"""CLI options are derived from the same schema used by the web API."""

import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click

from .engine import DEFAULT_MODEL, doctor as diagnose
from .jobs import JobManager, TERMINAL
from .schema import PRESETS, SongRequest
from .audio import ensure_mp3, export_mp3


@click.group()
def app():
    """MiniMax Music 3: local music generation and browser studio."""


@app.command()
@click.option("--model-path", type=click.Path(path_type=Path), default=DEFAULT_MODEL, show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default="outputs", show_default=True)
@click.option("--host", type=click.Choice(["127.0.0.1", "localhost", "::1"]), default="127.0.0.1")
@click.option("--port", type=click.IntRange(1, 65535), default=7860, show_default=True)
def serve(model_path, output_dir, host, port):
    """Open the local studio at http://127.0.0.1:7860."""
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(model_path, output_dir), host=host, port=port)


@app.command()
@click.option("--model-path", type=click.Path(path_type=Path), default=DEFAULT_MODEL, show_default=True)
def doctor(model_path):
    """Check local checkpoint completeness, dependencies, and CUDA."""
    report = diagnose(model_path)
    click.echo(json.dumps(report, indent=2))
    if not report["ready"]:
        raise click.exceptions.Exit(1)


@app.command()
@click.argument("name", type=click.Choice(list(PRESETS)))
def preset(name):
    """Print an editable song JSON request (redirect to a file)."""
    click.echo(SongRequest(**PRESETS[name]).model_dump_json(indent=2))


@app.command()
def schema():
    """Print the complete JSON request schema and numeric limits."""
    click.echo(json.dumps(SongRequest.model_json_schema(), indent=2))


@app.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default="outputs")
def history(output_dir):
    """List saved generations as JSON."""
    manager = JobManager(DEFAULT_MODEL, output_dir)
    try:
        click.echo(json.dumps(manager.list(), indent=2))
    finally:
        manager.close()


@app.command()
@click.argument("job_id")
@click.option("--output-dir", type=click.Path(path_type=Path), default="outputs")
def status(job_id, output_dir):
    """Read a generation's progress, settings, and audio filenames."""
    manager = JobManager(DEFAULT_MODEL, output_dir)
    try:
        click.echo(json.dumps(manager.get(job_id), indent=2))
    except KeyError as exc:
        raise click.ClickException("Generation not found") from exc
    finally:
        manager.close()


@app.command()
@click.argument("job_id")
@click.option("--server", default="http://127.0.0.1:7860", show_default=True)
def cancel(job_id, server):
    """Cancel a job owned by a running Studio server. Use Ctrl+C for a direct CLI job."""
    import re
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise click.ClickException("Invalid generation ID")
    request = Request(f"{server.rstrip('/')}/api/jobs/{job_id}/cancel", data=b"{}",
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=15) as response:
            click.echo(response.read().decode())
    except HTTPError as exc:
        raise click.ClickException(exc.read().decode()) from exc
    except (URLError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@app.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def preview(config):
    """Validate a request and print the exact caption and tagged lyrics."""
    try:
        song = SongRequest.model_validate_json(config.read_text())
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({"caption": song.caption(), "lyrics": song.rendered_lyrics()}, indent=2))


def song_options(function):
    """Expose every scalar request field; section arrays are supplied as JSON."""
    properties = SongRequest.model_json_schema()["properties"]
    for name, prop in reversed(list(properties.items())):
        if name == "sections":
            continue
        flag = "--" + name.replace("_", "-")
        kind = prop.get("type") or prop["anyOf"][0]["type"]
        help_text = prop.get("description", prop["title"])
        help_text += f" (default: {prop.get('default')!r})"
        kwargs = dict(default=None, help=help_text)
        if kind == "boolean":
            flag += "/--" + ("fixed-seed" if name == "randomize_seed" else "no-" + name.replace("_", "-"))
        elif "enum" in prop:
            kwargs["type"] = click.Choice(prop["enum"])
        elif kind == "integer":
            kwargs["type"] = click.IntRange(prop.get("minimum"), prop.get("maximum"))
        elif kind == "number":
            kwargs["type"] = click.FloatRange(prop.get("minimum"), prop.get("maximum"))
        else:
            kwargs["type"] = str
        function = click.option(flag, **kwargs)(function)
    return function


@app.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Song JSON; explicit flags override it")
@click.option("--preset", "preset_name", type=click.Choice(list(PRESETS)))
@click.option("--lyrics-file", type=click.Path(exists=True, path_type=Path))
@click.option("--sections-file", type=click.Path(exists=True, path_type=Path), help="JSON array of {name, lyrics, direction}")
@click.option("--model-path", type=click.Path(path_type=Path), default=DEFAULT_MODEL, show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default="outputs", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), help="Also save the final .wav or .mp3 here, with a .json request sidecar")
@click.option("--force", is_flag=True, help="Allow replacing --output and its sidecar")
@click.option("--dry-run", is_flag=True, help="Validate and print the assembled request without loading the model")
@click.option("--play", is_flag=True, help="Play decoded audio after headroom is buffered (requires ffplay)")
@song_options
def generate(config, preset_name, lyrics_file, sections_file, model_path, output_dir, output, force, dry_run, play, **overrides):
    """Generate music using local weights. Every Studio setting has a flag."""
    try:
        data = dict(PRESETS[preset_name]) if preset_name else {}
        if config:
            data.update(json.loads(config.read_text()))
        data.update({key: value for key, value in overrides.items() if value is not None})
        if lyrics_file:
            data.update(lyrics=lyrics_file.read_text(), sections=[])
        if sections_file:
            data.update(sections=json.loads(sections_file.read_text()), lyrics="")
        song = SongRequest.model_validate(data)
    except (ValueError, OSError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if dry_run:
        click.echo(json.dumps({"request": song.model_dump(), "caption": song.caption(), "lyrics": song.rendered_lyrics()}, indent=2))
        return
    if output:
        if output.suffix.lower() not in {".wav", ".mp3"}:
            raise click.ClickException("--output must end in .wav or .mp3")
        if not force and (output.exists() or output.with_suffix(".json").exists()):
            raise click.ClickException("Output already exists; choose a new path or pass --force")
        if output.suffix.lower() == ".mp3" and not shutil.which("ffmpeg"):
            raise click.ClickException("MP3 export requires FFmpeg; install it before generating an MP3")
    if play and not shutil.which("ffplay"):
        raise click.ClickException("--play requires ffplay (install FFmpeg)")
    manager = JobManager(model_path, output_dir)
    player = None
    played = 0
    try:
        job = manager.submit(song)
        click.echo(f"Generation {job['id']} • seed {job['seed']} • {song.steps} flow-matching steps", err=True)
        last_message = ""
        while True:
            job = manager.get(job["id"])
            if job["message"] != last_message:
                click.echo(job["message"], err=True)
                last_message = job["message"]
            if play and job.get("preview_ready"):
                import soundfile as sf
                if player is None:
                    player = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
                        "-f", "s16le", "-ar", str(job["sample_rate"]), "-ac", "2", "-i", "pipe:0"], stdin=subprocess.PIPE)
                for chunk in job["chunks"][played:]:
                    pcm, _ = sf.read(manager.directory(job["id"]) / chunk, dtype="int16", always_2d=True)
                    player.stdin.write(pcm.astype("<i2").tobytes())
                    player.stdin.flush()
                    played += 1
            if job["stage"] in TERMINAL:
                break
            time.sleep(0.5)
        if job["stage"] != "complete":
            raise click.ClickException(f"{job['message']}\nLog: {manager.directory(job['id']) / 'worker.log'}")
        audio = manager.directory(job["id"]) / "song.wav"
        if output:
            if output.suffix.lower() == ".mp3":
                try:
                    audio = ensure_mp3(audio)
                except RuntimeError as exc:
                    raise click.ClickException(str(exc)) from exc
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio, output)
            shutil.copyfile(audio.with_name("request.json"), output.with_suffix(".json"))
        click.echo(str(output.resolve() if output else audio))
    except (KeyboardInterrupt, BrokenPipeError):
        raise click.ClickException("Generation or playback cancelled")
    finally:
        manager.close()
        if player:
            try:
                player.stdin.close()
            except BrokenPipeError:
                pass
            try:
                player.wait(timeout=5 if job["stage"] != "complete" else 360)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                player.terminate()


@app.command("export")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), help="MP3 destination; defaults to the WAV name with .mp3")
@click.option("--force", is_flag=True, help="Replace an existing MP3")
def export_audio(source, output, force):
    """Convert an existing WAV to 320 kbps MP3 without generating music again."""
    output = output or source.with_suffix(".mp3")
    if source.suffix.lower() != ".wav" or output.suffix.lower() != ".mp3":
        raise click.ClickException("Export requires a .wav source and .mp3 destination")
    try:
        export_mp3(source, output, overwrite=force)
    except (RuntimeError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(output.resolve()))


if __name__ == "__main__":
    app()
