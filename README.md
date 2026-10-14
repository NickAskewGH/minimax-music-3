# MiniMax Music 3 · Local Studio

A local browser studio and CLI for the downloaded MiniMax Music 3 model. Both use the same validated song settings and inference code. Features include a section chart, tagged lyrics, structured captions, all official Studio generation controls, live monitoring, cancellation, and a saved song library.

## Start

Requires Linux, Python 3.12+, uv, and an NVIDIA GPU with a working CUDA driver. Dependencies are locked in `uv.lock`; the inference extra includes CUDA PyTorch.

MP3 export also uses system **FFmpeg with libmp3lame** (on Ubuntu/Debian: `sudo apt install ffmpeg`). Without it, generation still saves the WAV and reports that MP3 export is unavailable.

```sh
uv sync --locked --extra inference
uv run --extra inference music3 doctor
uv run --extra inference music3 serve
```

Open **http://127.0.0.1:7860**. The default checkpoint is `~/Downloads/minimax-music-3` (your existing download). Override it with `--model-path /path/to/model` or `MUSIC3_MODEL_PATH`. Weights are loaded from that directory with offline mode enabled; no replacement model is downloaded.

The task runner is installed through uv as `rust-just`:

```sh
uv run --extra inference just studio
uv run --extra inference just doctor
uv run --extra inference just test
```

The server binds to loopback only. `--port` changes the default 7860; `--output-dir` changes the default `outputs/`. Ctrl+C stops the server and cancels its jobs. Run on the host, or expose the NVIDIA device to your container. A sandbox without GPU access can display the UI but cannot generate audio.

## Write a song

Choose a starter or supply your own genre, subgenre, tempo, key, scale, emotional progression, vocal character, instruments, and production details. Each category also has a free-text field for unrestricted directions.

In **Section chart**, add, remove, or reorder sections. Each has a name, sung lyrics, and arrangement directions. Directions are assembled into the caption; only lyrics go into the lyrics input. Alternatively, choose **Tagged lyrics**:

```text
[verse]
Morning spills across the floor
I leave my worries by the door

[chorus]
Let the daylight carry me
I like squashing a little pea
```

Put tags on separate lines: the checkpoint drops text after a leading tag. The app rejects that input instead of silently losing words. Instrumental mode sends `[instrumental]` and retains section arrangement directions. The model limits assembled text to 5,000 tokens. Tempo, key, lyrics, and arrangement are generative instructions, not exact constraints.

**View prompt** shows the caption and lyrics before inference. **Export JSON** and **Import JSON** move settings between the web UI and CLI. The current draft is saved in this browser; generated requests and audio are saved on disk.

## Generation controls

| Studio control | CLI option | Range / default |
| --- | --- | --- |
| Maximum duration | `--duration` | 5–300 seconds; 60; may end earlier |
| Seed | `--seed` | 0–2,147,483,647; 0 |
| Randomize seed | `--randomize-seed` / `--fixed-seed` | Random by default; actual seed saved |
| Playback headroom | `--headroom` | 0–60 seconds; 0 |
| Flow-matching steps per chunk | `--steps` | 4–60; 30 recommended for listening |
| Guidance scale | `--guidance` | 1–4; 1.7 |
| Memory mode | `--offload` | `model` (default), `none`, `sequential` |

Keep the default 30 steps for listening. Very low values such as 4 are useful for speed/resource tests but can severely degrade the sound. A valid 44.1 kHz WAV does not guarantee that the model rendered good audio.

`model` moves the language model and depth decoder back to CPU before loading acoustic models on the GPU. This is the default for the 24 GB RTX 3090. `none` retains all components on the GPU and needs more VRAM; it exhausted memory on this 24 GB desktop, so use the default `model` mode here. `sequential` streams language-model layers from CPU and uses less VRAM at a substantial speed cost. Long prompts and songs also need more KV-cache memory. If a job runs out of memory, use more offloading, shorten the prompt or duration, or free VRAM.

The first phase composes musical frames; the second renders audio windows. Live monitoring starts during the second phase once the selected headroom is available. If the song is shorter than the headroom, playback starts when rendering finishes. During a slow render, playback can temporarily exhaust buffered audio. The final WAV joins the original windows without adding those gaps. Headroom changes playback timing, not the waveform.

The downloaded Diffusers vocoder declares **44,100 Hz**. The app reads the checkpoint sample rate and writes stereo PCM-16 WAVs. The [upstream documentation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_music3.md) confirms that 44.1 kHz is native and the reference server resamples it to 32 kHz.

## CLI

```sh
# Generate from an editable section-chart example.
uv run --extra inference music3 generate \
  --config examples/daybreak.json \
  --duration 60 --fixed-seed --seed 7 \
  --steps 30 --guidance 1.7 --output outputs/daybreak.wav

# A full structured song (three verses, choruses, bridge and solo).
uv run --extra inference music3 generate --config examples/full-song.json

# Or start directly with a preset.
uv run --extra inference music3 generate \
  --preset after-hours --duration 120 --offload model

# Supply tagged lyrics and a structured caption.
uv run --extra inference music3 generate \
  --lyrics-file lyrics.txt --genre Soul --bpm 88 --key D --scale minor \
  --vocal-gender Female --timbre 'Warm alto' \
  --instruments 'Rhodes, upright bass, brushed drums' \
  --arrangement '[verse] Sparse piano. [chorus] Bass and drums enter.'

# Preview without loading weights or requiring a GPU.
uv run --extra inference music3 generate --config examples/daybreak.json --dry-run
uv run --extra inference music3 preview examples/daybreak.json

# All flags, JSON schema, and a complete preset request.
uv run --extra inference music3 generate --help
uv run --extra inference music3 schema
uv run --extra inference music3 history
uv run --extra inference music3 status GENERATION_ID
uv run --extra inference music3 cancel GENERATION_ID
uv run --extra inference music3 preset neon > neon.json
```

Every scalar JSON setting has a matching CLI flag (underscores become hyphens). Use `--sections-file chart.json` for an array of `{name, lyrics, direction}`. Use raw lyrics or a chart, rather than both. Flags override `--config`, and config values override `--preset`. To override a preset's chart with lyrics, use `--lyrics-file`, which clears its sections.

`--play` enables incremental CLI playback through `ffplay` (install FFmpeg if needed). `--headroom` applies to playback. Without `--play`, generation is silent. Ctrl+C cancels the worker and releases GPU memory. `--output` optionally copies the finished `.wav` or `.mp3` and a matching `.json` request sidecar to the chosen location; existing files require `--force` to replace. The job directory retains the WAV master and MP3 copy.

Every new song also gets a **320 kbps MP3**, preserving the native **44.1 kHz stereo** sample rate. MP3 is lossy; keep the WAV for editing or archiving. The Studio offers **Download WAV** and **Download MP3**. MP3 downloads also work for older completed songs: the server creates and caches the MP3 from their existing WAV, without running the model again.

```sh
# Generate a song and copy its MP3 to a chosen path (WAV stays in the job directory).
uv run --extra inference music3 generate --config examples/daybreak.json --output outputs/daybreak.mp3

# Export an existing WAV, without loading the model. Output defaults to the same name with .mp3.
uv run --extra inference music3 export outputs/the-light-we-carry-30-steps.wav
```

`music3 export` accepts `--output path.mp3` and `--force`; it leaves the source WAV untouched.

## Saved generations and API

Each generation gets an `outputs/<id>/` directory containing:

- `song.wav`: finished music.
- `song.mp3`: 320 kbps listening/sharing copy, when FFmpeg is available.
- `request.json`: complete settings with actual seed and randomization disabled.
- `prompt.txt`: assembled caption and lyrics.
- `status.json`: progress, timing, sample rate, and final state.
- `chunk-*.wav`: rendered preview windows.
- `worker.log`: model loading and inference diagnostics.

One GPU worker runs at a time across CLI and web processes. The Studio queues up to eight requests. Cancellation terminates the worker and releases GPU memory. Jobs from another running process are visible but must be cancelled in that process. Finished audio survives restarts; interrupted jobs never restart automatically. Choose **Use these settings** to make another version.

Local API documentation is at **http://127.0.0.1:7860/docs**. Principal endpoints: `POST /api/preview`, `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, and `GET /api/jobs/{id}/files/song.wav` (or `song.mp3`).

## Implementation and checks

The inference adapter uses the official Diffusers text encoding, autoregressive generation, and chunk-denoising blocks. It changes component placement between phases and exposes decoded windows for playback. The language-model and acoustic stages use separate RNGs seeded with `seed` and `seed + 1`, matching the Studio convention. Identical seeds are reproducible within the same runtime and hardware; library or device changes may affect results.

Diffusers is pinned to the official Studio's commit `82319140e0456fd58beff0a251c38825bfc310de`. This experimental pipeline uses some upstream internal interfaces, so upgrading Diffusers should include a real GPU generation check. Add dependencies with `uv add` or `uv add --dev` and commit both `pyproject.toml` and `uv.lock`.

```sh
uv run --extra inference pytest
node --check src/music3/static/app.js
```

Tests cover prompt/lyrics separation, parameter bounds, seed persistence, CLI parity and overrides, checkpoint integrity, queueing, cancellation, failure, restart history, API validation, and file downloads. Worker lifecycle tests use a tiny test subprocess. Real GPU checks are separate; see `VALIDATION.md` for recorded results.

The app does not call an external language model to rewrite prompts or invent lyrics. Its structured editor and presets work locally. The model's license remains in your checkpoint's `LICENSE` file.

References: [MiniMax model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3), [official Studio source and controls](https://huggingface.co/spaces/MiniMaxAI/MiniMax-Music3/blob/main/app.py), [pinned Diffusers implementation](https://github.com/huggingface/diffusers/tree/82319140e0456fd58beff0a251c38825bfc310de/src/diffusers/modular_pipelines/minimax_music3).
