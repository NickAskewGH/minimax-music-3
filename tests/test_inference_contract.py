"""Check the installed upstream input contract without loading model weights."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.modular_pipelines.minimax_music3.encoders import MiniMaxMusic3TextEncoderStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState


def test_upstream_rejects_over_5000_tokens():
    components = SimpleNamespace(tokenizer=lambda *args, **kwargs: {"input_ids":torch.zeros((1,5001), dtype=torch.long)},
                                 _execution_device=torch.device("cpu"))
    state = PipelineState()
    state.set("prompt", "Acoustic pop")
    state.set("lyrics", "[verse]\nTest words")
    with pytest.raises(ValueError, match="maximum is 5000"):
        MiniMaxMusic3TextEncoderStep()(components, state)


def test_upstream_prompt_special_tokens_and_cfg_rows():
    captured = {}
    def tokenize(text, **kwargs):
        captured["text"] = text
        return {"input_ids":torch.tensor([[10,20,30,40,50,60]])}
    components = SimpleNamespace(tokenizer=tokenize, _execution_device=torch.device("cpu"))
    state = PipelineState()
    state.set("prompt", "Genre: acoustic pop.")
    state.set("lyrics", "[Verse]\nMorning light")
    _, result = MiniMaxMusic3TextEncoderStep()(components, state)
    assert captured["text"] == '<|im_start|><|caption_start|>Genre: acoustic pop.<|caption_end|><|lyrics_start|>[start]\n[verse]\nMorning light<|lyrics_end|><|im_end|><|audio_start|>'
    assert result.text_ids[0].tolist() == [10,20,30,40,50,60]
    assert result.text_ids[1].tolist() == [10,151654,151654,151654,50,60]
