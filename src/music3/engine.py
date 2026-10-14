"""Local inference using the pinned upstream MiniMax modular blocks.

We run the official blocks in phases to keep the language model and acoustic
models out of VRAM at the same time. Decoded windows are published incrementally.
"""

import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace

from .schema import SongRequest
from .audio import ensure_mp3

DEFAULT_MODEL = Path(os.environ.get("MUSIC3_MODEL_PATH", str(Path.home() / "Downloads/minimax-music-3")))
COMPONENTS = ("condition_encoder", "language_model", "rvq_depth_decoder", "scheduler", "tokenizer", "transformer", "vocoder")


def inspect_model(path: Path) -> dict:
    """Check completeness without loading weights or downloading replacements."""
    path = path.expanduser().resolve()
    issues = []
    if not (path / "modular_model_index.json").is_file():
        issues.append("Missing modular_model_index.json")
    for name in COMPONENTS:
        folder = path / name
        config = "scheduler_config.json" if name == "scheduler" else "tokenizer_config.json" if name == "tokenizer" else "config.json"
        if not (folder / config).is_file():
            issues.append(f"Missing {name}/{config}")
        if name == "scheduler":
            continue
        if name == "tokenizer":
            if not (folder / "tokenizer.json").is_file():
                issues.append("Missing tokenizer/tokenizer.json")
            continue
        weights = list(folder.glob("*.safetensors"))
        if not weights:
            issues.append(f"Missing {name} safetensors weights")
        for index in folder.glob("*.safetensors.index.json"):
            try:
                for shard in set(json.loads(index.read_text())["weight_map"].values()):
                    if not (folder / shard).is_file():
                        issues.append(f"Missing {name}/{shard}")
            except (ValueError, KeyError) as exc:
                issues.append(f"Invalid weight index {index.name}: {exc}")
        for weight in weights:
            try:
                with weight.open("rb") as f:
                    header_size = struct.unpack("<Q", f.read(8))[0]
                    if header_size > 100_000_000:
                        raise ValueError("not a safetensors file (possibly a download pointer)")
                    header = json.loads(f.read(header_size))
                data_size = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
                if weight.stat().st_size != 8 + header_size + data_size:
                    raise ValueError("incomplete or oversized download")
            except (ValueError, KeyError, struct.error) as exc:
                issues.append(f"Invalid {name}/{weight.name}: {exc}")
    return {"model_path": str(path), "model_ready": not issues, "issues": issues}


def doctor(path: Path = DEFAULT_MODEL) -> dict:
    report = inspect_model(path)
    report["inference_installed"] = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "transformers", "accelerate"))
    report["cuda_available"] = False
    if importlib.util.find_spec("torch"):
        try:
            import torch
            report["torch_version"] = torch.__version__
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                report["gpu"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                report.update(vram_free_gib=round(free / 2**30, 2), vram_total_gib=round(total / 2**30, 2))
        except Exception as exc:
            report["issues"].append(str(exc))
    if not report["inference_installed"]:
        report["issues"].append("Install the inference runtime: uv sync --extra inference")
    if not report["cuda_available"]:
        report["issues"].append("CUDA is unavailable. Run on the host with an NVIDIA driver and GPU access.")
    report["ready"] = not report["issues"]
    return report


def generate(request: SongRequest, model_path: Path, output: Path, emit):
    """Write song.wav and independent preview chunks; emit JSON-compatible progress."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    report = doctor(model_path)
    if not report["ready"]:
        raise RuntimeError("\n".join(report["issues"]))
    import numpy as np
    import soundfile as sf
    import torch
    from diffusers import ClassifierFreeGuidance, ModularPipeline
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from diffusers.modular_pipelines.minimax_music3.decoders import _CROP_LEFT_LATENT, _CROP_RIGHT_LATENT

    # Too many CPU threads makes the tiny per-frame operations much slower.
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    emit({"stage": "loading", "message": "Loading local model weights", "progress": 0.01,
          "runtime": {name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate")},
          "diffusers_commit": "82319140e0456fd58beff0a251c38825bfc310de"})
    pipe = ModularPipeline.from_pretrained(str(model_path), local_files_only=True)
    # This is a Qwen tokenizer. Transformers misidentifies the local root config
    # as Mistral when transformers_version is absent; preserve the shipped regex.
    pipe.load_components(dtype=torch.bfloat16, local_files_only=True,
                         pretrained_model_name_or_path=str(model_path),
                         fix_mistral_regex={"tokenizer": False})
    missing = [name for name in COMPONENTS if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"Failed to load local components: {', '.join(missing)}. See worker.log.")
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=request.guidance))
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval()

    # Explicit device facade avoids inferring the execution device from an offloaded component.
    components = SimpleNamespace(**pipe.components, _execution_device=torch.device("cuda"),
        frame_rate=pipe.frame_rate, num_codebooks=pipe.num_codebooks,
        audio_vocab_size=pipe.audio_vocab_size, num_channels_latents=pipe.num_channels_latents)
    state = PipelineState()
    for key, value in {"prompt": request.caption(), "lyrics": request.rendered_lyrics(),
                       "audio_duration": request.duration, "num_inference_steps": request.steps,
                       "generator": torch.Generator("cuda").manual_seed(request.seed)}.items():
        state.set(key, value)

    blocks = pipe.blocks.sub_blocks
    with torch.inference_mode():
        _, state = blocks["text_encoder"](components, state)
        emit({"stage": "semantic", "message": "Composing musical frames", "progress": 0.04,
              "prompt_tokens": state.text_ids.shape[1], "sample_rate": pipe.sampling_rate})
        if request.offload == "none":
            pipe.to("cuda")
        else:
            pipe.rvq_depth_decoder.to("cuda")
            if request.offload == "sequential":
                from diffusers.hooks import apply_group_offloading
                apply_group_offloading(pipe.language_model, onload_device=torch.device("cuda"),
                                       offload_type="leaf_level", use_stream=True)
            else:
                pipe.language_model.to("cuda")

        counter = 0
        max_frames = int(request.duration * pipe.frame_rate)

        def progress_hook(module, args, result):
            nonlocal counter
            counter += 1
            if counter % 25 == 0:
                emit({"stage": "semantic", "message": f"Composed {min(counter, max_frames) / pipe.frame_rate:.1f}s of musical frames",
                      "progress": 0.04 + 0.65 * min(counter / max_frames, 1)})

        hook = pipe.language_model.model.register_forward_hook(progress_hook)
        try:
            _, state = blocks["semantic_generator"](components, state)
        finally:
            hook.remove()
        if request.offload != "none":
            if request.offload != "sequential":
                pipe.language_model.to("cpu")
            pipe.rvq_depth_decoder.to("cpu")
            torch.cuda.empty_cache()
            for name in ("condition_encoder", "transformer", "vocoder"):
                getattr(pipe, name).to("cuda")

        _, state = blocks["prepare_chunks"](components, state)
        denoiser = blocks["denoise"]
        chunk_state = denoiser.get_block_state(state)
        chunk_state.latent_chunks = []
        chunk_state.previous_latent = None
        chunk_state.previous_condition = None
        # Same independent denoising RNG convention as the official Studio.
        chunk_state.generator = torch.Generator("cuda").manual_seed(request.seed + 1)
        chunks = []
        count = len(chunk_state.chunk_starts)
        sample_rate = pipe.sampling_rate
        for index in range(count):
            emit({"stage": "decoding", "message": f"Rendering audio window {index + 1} of {count}",
                  "progress": 0.70 + 0.29 * index / count})
            _, chunk_state = denoiser.loop_step(components, chunk_state, k=index)
            latents = chunk_state.latent_chunks.pop()
            waveform = pipe.vocoder(latents.to(pipe.vocoder.dtype))
            left = 0 if index == 0 else _CROP_LEFT_LATENT * pipe.latent_hop_length
            right = 0 if index == count - 1 else _CROP_RIGHT_LATENT * pipe.latent_hop_length
            audio = waveform[0, :, left:waveform.shape[-1] - right].float().clamp(-1, 1).cpu().numpy().T
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("The model produced empty or non-finite audio")
            chunks.append(audio)
            chunk_name = f"chunk-{index:04d}.wav"
            sf.write(output / chunk_name, audio, sample_rate, subtype="PCM_16")
            seconds = sum(len(c) for c in chunks) / sample_rate
            emit({"stage": "decoding", "chunk": chunk_name, "sample_rate": sample_rate,
                  "audio_seconds": seconds, "preview_ready": seconds >= request.headroom or index == count - 1})

        final = np.concatenate(chunks)
        sf.write(output / "song.wav", final, sample_rate, subtype="PCM_16")
        emit({"stage": "exporting", "message": "Encoding 320 kbps MP3", "progress": 0.99})
        exports = {}
        message = "Music ready"
        try:
            exports["mp3"] = ensure_mp3(output / "song.wav").name
        except RuntimeError as exc:
            # A missing encoder must never discard a successful, expensive generation.
            exports["mp3_error"] = str(exc)
            message = f"WAV ready; {exc}"
        emit({"stage": "complete", "message": message, "progress": 1, **exports,
              "audio_seconds": len(final) / sample_rate, "sample_rate": sample_rate,
              "elapsed_seconds": round(time.monotonic() - started, 2), "audio": "song.wav"})
