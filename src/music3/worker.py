"""Isolated GPU worker. JSON on stdout; library diagnostics on stderr."""

import contextlib
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from .engine import generate
from .schema import SongRequest


def main():
    directory = Path(sys.argv[1])
    model = Path(sys.argv[2])
    request = SongRequest.model_validate_json((directory / "request.json").read_text())
    stream = sys.stdout

    def emit(event):
        print(json.dumps(event), file=stream, flush=True)

    try:
        # Coordinate GPU use between separate CLI and web processes as well.
        with open(f"/tmp/music3-gpu-{os.getuid()}.lock", "a") as gpu_lock, contextlib.redirect_stdout(sys.stderr):
            emit({"stage": "waiting", "message": "Waiting for exclusive GPU access"})
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            generate(request, model, directory, emit)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        message = str(exc)
        if type(exc).__name__ == "OutOfMemoryError":
            message = "GPU memory exhausted. Select CPU offload (or layer offload), reduce duration, or free GPU memory. Full details are in worker.log."
        emit({"stage": "failed", "message": message})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
