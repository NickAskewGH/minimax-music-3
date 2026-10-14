# Validation record — 14 October 2026

Hardware: NVIDIA GeForce RTX 3090, 24 GB VRAM; 128 GB system RAM. Runtime: Python 3.12.7, PyTorch 2.14.1+cu130, Transformers 5.19.0, Diffusers commit `82319140e0456fd58beff0a251c38825bfc310de`.

## Interface parity

The official Studio source was inspected, including its `generate()` inputs and advanced controls. Implemented in both browser and CLI:

| Official Studio input | Local UI / CLI |
| --- | --- |
| Lyrics, global metadata, vocal details, arrangement | Tagged lyrics or section chart, free-text fields, and individual musical attributes; all available as CLI flags or JSON |
| Maximum duration (5–300 s) | Duration slider / `--duration` |
| Seed and randomization | Seed and checkbox / `--seed`, `--randomize-seed`, `--fixed-seed` |
| Playback headroom (0–60 s) | Live monitoring / `--headroom` with `--play` |
| Flow-matching steps (4–60) | Advanced controls / `--steps` |
| Guidance (1–4) | Advanced controls / `--guidance` |

The Space's external LLM songwriting assistant and share-video renderer are not model inference settings and are not included. The local editor accepts a complete caption and lyrics without external services. Extra local controls cover memory offloading, model/output paths, and server port.

## Automated checks

- `pytest -q`: **34 passed**. Includes real worker subprocess lifecycle tests with a lightweight fixture, CLI option parity, JSON validation and round trips, queue cancellation and failure, persistence, API downloads, checkpoint integrity detection, and MP3 export tests.
- The installed upstream text encoder rejects prompts over 5,000 tokens and constructs the expected special tokens and classifier-free guidance rows.
- The local Qwen tokenizer produces identical IDs to the downloaded `tokenizer.json` for English, Chinese, and model special tokens. The unrelated Mistral-regex patch is explicitly disabled.
- `node --check src/music3/static/app.js`: passed.
- `uv lock --check`: passed.
- A wheel was built and inspected: all Python modules and HTML/CSS/JS assets are included.
- The CLI's FFmpeg PCM playback pipeline completed successfully using SDL's dummy audio driver, without playing audio aloud.

The test runner reports one upstream Starlette deprecation warning about its HTTPX test-client adapter; there are no failing checks.

## Real local GPU generations

These checks establish execution, file integrity, timing and reproducibility; they do not establish perceptual audio quality. The user reported severe audible degradation in the 4-step `the-light-we-carry.wav`. That file was a duration/resource test and should not have been presented as a finished listening example. Use the normal 30-step setting for listening; the 4-step tests below remain useful only as engineering checks.

### Audio-quality investigation

Re-rendered the same full-song request with seed 23, changing only `steps` from 4 to 30. The first eight windows (33.042 seconds) show a substantial spectral and stereo difference:

| Measurement over the same opening | 4 steps | 30 steps |
| --- | --- | --- |
| Energy below 1 kHz | 3.25% | 83.4% |
| Energy between 1 and 5 kHz | 93.8% | 16.0% |
| Left/right correlation | −0.211 | 0.590 |
| RMS amplitude | 0.0526 | 0.0932 |

Spectral energy was measured using 8,192-sample Hann-windowed FFTs, averaged across non-overlapping frames and both channels, normalized by total energy. These measurements support the reported thin, midrange-heavy sound and implicate the 4-step render. They are not a substitute for listening or a general quality score. The comparison preview is `outputs/the-light-we-carry-30-steps-preview.wav`; it contains the unmodified first eight 30-step PCM windows.

The user listened to this preview and confirmed it was **noticeably better**. This validates the improvement for this sample; it does not imply every generated song will have the same quality.

The completed 30-step render is `outputs/the-light-we-carry-30-steps.wav`, with the exact request in its `.json` sidecar: **295.509 seconds**, 73 windows, stereo 44.1 kHz PCM-16, **949.67 seconds** inference time. Its duration matches the original exactly. The final WAV matches every rendered window sample-for-sample, and its opening matches the preview the user heard. All samples are finite and nonzero audio is present. The original 4-step file is preserved for comparison.

The guidance formula, denoising schedule, chunk crops and PCM export were checked against the official Studio source. The [Diffusers documentation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_music3.md) confirms native 44.1 kHz output; the reference server separately resamples to 32 kHz. No resampling, EQ or other audio post-processing was added as a workaround.

The following used the downloaded checkpoint with offline loading, not a substitute model or generated test signal:

| Test | Actual output | Result |
| --- | --- | --- |
| CLI, seed 7, 5 s limit, 4 steps, default CPU offload | 4.992 s stereo WAV | Passed; 27.55 s inference time |
| Browser, seed 0, 15 s limit, 30 steps, headroom 5 s | 15.000 s, 3 windows | Passed; 61.13 s inference time |
| Layer offload, seed 17, 5 s limit, guidance 1, headroom 60 s | 4.992 s stereo WAV | Passed; 430.63 s inference time; about 4.8 GB total GPU memory observed during composition |
| Instrumental, 300 s limit, 4 steps | 33.913 s, 8 windows | Passed; model emitted its end token before the maximum; 88.21 s inference time |
| Browser live monitoring, seed 0, 10 s limit, 4 steps | 9.996 s stereo WAV | Passed; 37.69 s inference time; no browser errors |

Output files are 44,100 Hz stereo PCM-16, matching this Diffusers vocoder's config. Checked outputs contain finite, nonzero samples. The 15-second final WAV is sample-for-sample equal to its concatenated preview files.

Keeping all components on the GPU (`--offload none`) exhausted VRAM on this desktop. That mode is available for machines with more free VRAM; the verified default here is `model`. Failure released the GPU, and subsequent generations succeeded. The downloaded checkpoint was not modified.

Full structured song: `examples/full-song.json` with `--steps 4`, seed 23, CPU offload and a 300-second maximum produced **295.509 seconds (4:55.5)** of 44.1 kHz stereo PCM-16 audio across **73 windows**, in **679.63 seconds**. All samples are finite, and all preview windows exactly match the final waveform. The model ended naturally just before the cap. Output: `outputs/the-light-we-carry.wav`; actual parameters: `outputs/the-light-we-carry.json`. The example file defaults to 30 steps for normal use; this long-duration resource check used 4 steps.

Repeat-seed check: two independent CLI workers using seed 7, 5 seconds, 4 steps and CPU offload produced **byte-identical WAVs**. SHA-256: `8c574bde099214bed65409d30126b3d373231a330d9d239856eb89c86552a1fa`. Files: `outputs/smoke.wav` and `outputs/reproducibility.wav`.

## MP3 export checks

FFmpeg/libmp3lame encoded `outputs/the-light-we-carry-30-steps.wav` to `outputs/the-light-we-carry-30-steps.mp3` without regenerating music. ffprobe confirmed MP3, 320,000 bits/s, 44,100 Hz stereo, and 11,823,064 bytes. The container duration is 295.549 seconds (MP3 frame padding); the source WAV is 295.509 seconds.

Automated tests exercise actual MP3 encoding and decoding, sample count and stereo preservation on a known test signal, overwrite protection, failed-export cleanup, missing FFmpeg handling, cached MP3 downloads for older completed jobs, and CLI generation output with a lightweight worker fixture. The source WAV is verified unchanged. New-generation MP3 creation uses the same tested encoder; no additional GPU generation was needed for this export change. JavaScript syntax and Python compilation checks pass. The new browser download link is covered by its API path but has not been clicked in a live browser during this change.

## Browser checks

Verified in Chrome: layout, hardware readiness indicator, advanced settings, real generation, saved generation reload, playable audio, live monitoring without console errors, cancelling an active worker, cancelling a queued Studio job from the CLI, section reordering, section-chart to tagged-lyrics conversion, assembled prompt preview, and reusing saved settings. Other active CLI processes appear with live state rather than being falsely marked interrupted.

Browser JSON import could not be exercised: the browser security layer rejected the local file selection because upload permission was declined. No workaround was attempted. Shared JSON validation, CLI file input, and API request/download round trips are tested. Browser file-selection behavior remains unverified.
