"""Persistent local job history and a single cancellable inference queue."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

from .schema import SongRequest

TERMINAL = {"complete", "failed", "cancelled", "interrupted"}


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class JobManager:
    def __init__(self, model_path: Path, output_dir: Path):
        self.model_path = model_path.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="music3")
        self.lock = threading.RLock()
        self.jobs = {}
        self.processes = {}
        self.closed = False

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError("Unknown generation")
        return self.output_dir / job_id

    def _save(self, job):
        folder = self.directory(job["id"])
        temporary = folder / "status.tmp"
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(folder / "status.json")

    def get(self, job_id):
        with self.lock:
            if job_id in self.jobs:
                return {**json.loads(json.dumps(self.jobs[job_id])), "can_cancel": True}
            file = self.directory(job_id) / "status.json"
            if not file.is_file():
                raise KeyError("Unknown generation")
            job = json.loads(file.read_text())
            owner = process_identity(job.get("owner_pid", 0))
            if job["stage"] not in TERMINAL and (not owner or owner != job.get("owner_start")):
                # Old sessions are never silently restarted.
                job.update(stage="interrupted", message="The server stopped before this generation finished")
            job["can_cancel"] = False
            return job

    def list(self):
        ids = {p.parent.name for p in self.output_dir.glob("*/status.json")}
        jobs = [self.get(job_id) for job_id in ids]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:100]

    def submit(self, request: SongRequest):
        with self.lock:
            if self.closed:
                raise RuntimeError("The studio is shutting down")
            if sum(j["stage"] not in TERMINAL for j in self.jobs.values()) >= 8:
                raise ValueError("The queue is full (8 songs); wait for a generation to finish")
            request = request.resolved()
            job_id = uuid.uuid4().hex
            folder = self.directory(job_id)
            folder.mkdir()
            (folder / "request.json").write_text(request.model_dump_json(indent=2))
            (folder / "prompt.txt").write_text(request.caption() + "\n\nLyrics\n" + request.rendered_lyrics())
            job = dict(id=job_id, title=request.title, seed=request.seed, stage="queued", message="Queued",
                       progress=0, chunks=[], preview_ready=False,
                       created_at=datetime.now(timezone.utc).isoformat(), request=request.model_dump(),
                       model_path=str(self.model_path), owner_pid=os.getpid(), owner_start=process_identity(os.getpid()))
            self.jobs[job_id] = job
            self._save(job)
            self.pool.submit(self._run, job_id)
            return self.get(job_id)

    def _run(self, job_id):
        folder = self.directory(job_id)
        with self.lock:
            job = self.jobs[job_id]
            if job["stage"] in TERMINAL:
                return
            job.update(stage="starting", message="Starting local inference")
            self._save(job)
        try:
            with (folder / "worker.log").open("w") as log:
                with self.lock:
                    if job["stage"] == "cancelled":
                        return
                    process = subprocess.Popen([sys.executable, "-m", "music3.worker", str(folder), str(self.model_path)],
                        stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
                    self.processes[job_id] = process
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        log.write(line)
                        continue
                    with self.lock:
                        if job["stage"] == "cancelled":
                            continue
                        if "chunk" in event:
                            job["chunks"].append(event.pop("chunk"))
                        job.update(event)
                        self._save(job)
                code = process.wait()
                with self.lock:
                    if job["stage"] not in TERMINAL:
                        job.update(stage="failed", message=f"Inference worker exited ({code}); see worker.log")
                    elif code and job["stage"] == "complete":
                        job.update(stage="failed", message=f"Inference worker exited unexpectedly ({code})")
                    self._save(job)
        except Exception as exc:
            with self.lock:
                if job["stage"] != "cancelled":
                    job.update(stage="failed", message=str(exc))
                    self._save(job)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["stage"] in TERMINAL:
                return job
            if job_id not in self.jobs:
                raise ValueError("This generation belongs to another running CLI or Studio process; cancel it there")
            job = self.jobs[job_id]
            job.update(stage="cancelled", message="Generation cancelled")
            self._save(job)
            process = self.processes.get(job_id)
            if process and process.poll() is None:
                process.terminate()
        if process:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return self.get(job_id)

    def close(self):
        with self.lock:
            self.closed = True
            ids = list(self.jobs)
        for job_id in ids:
            self.cancel(job_id)
        self.pool.shutdown(wait=True, cancel_futures=True)
