from contextlib import asynccontextmanager
from pathlib import Path
import re
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .engine import DEFAULT_MODEL, doctor
from .jobs import JobManager
from .schema import PRESETS, SongRequest
from .audio import ensure_mp3


def create_app(model_path: Path = DEFAULT_MODEL, output_dir: Path = Path("outputs")):
    manager = JobManager(model_path, output_dir)
    static = Path(__file__).parent / "static"

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="MiniMax Music 3 — Local Studio", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/system")
    def system():
        return doctor(manager.model_path)

    @app.get("/api/schema")
    def schema():
        return SongRequest.model_json_schema()

    @app.get("/api/presets")
    def presets():
        return {key: SongRequest(**value).model_dump() for key, value in PRESETS.items()}

    @app.post("/api/preview")
    def preview(song: SongRequest):
        return {"caption": song.caption(), "lyrics": song.rendered_lyrics()}

    @app.get("/api/jobs")
    def jobs():
        return manager.list()

    @app.post("/api/jobs", status_code=202)
    def generate(song: SongRequest):
        try:
            return manager.submit(song)
        except ValueError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Generation not found") from exc

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Generation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def file(job_id: str, filename: str):
        saved = job(job_id)
        if filename not in {"song.wav", "song.mp3", "request.json", "prompt.txt", "worker.log", "status.json"} and not re.fullmatch(r"chunk-\d{4}\.wav", filename):
            raise HTTPException(404, "File not found")
        path = manager.directory(job_id) / filename
        if filename == "song.mp3":
            if saved["stage"] != "complete":
                raise HTTPException(409, "Wait for the song to finish before exporting MP3")
            try:
                ensure_mp3(path.with_suffix(".wav"))
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
        if not path.is_file():
            raise HTTPException(404, "File not ready")
        return FileResponse(path, filename=filename, media_type="audio/mpeg" if filename == "song.mp3" else None)

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
