"""One validated request contract for the CLI, browser, and inference worker."""

import re
import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_SEED = 2**31 - 1
TAGS = ["intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "instrumental", "solo", "outro"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Section(StrictModel):
    name: str = Field(default="verse", min_length=1, max_length=80)
    lyrics: str = Field(default="", max_length=12000)
    direction: str = Field(default="", max_length=3000)

    @model_validator(mode="after")
    def valid_name(self):
        if any(c in self.name for c in "[]\n\r"):
            raise ValueError("Section names cannot contain brackets or line breaks")
        return self


class SongRequest(StrictModel):
    title: str = Field(default="Untitled session", max_length=160)
    prompt: str = Field(default="", max_length=30000, description="Free-form music description, added before the structured caption")
    global_metadata: str = Field(default="", max_length=12000)
    vocal_details: str = Field(default="", max_length=12000)
    arrangement: str = Field(default="", max_length=12000)
    genre: str = Field(default="", max_length=300)
    subgenre: str = Field(default="", max_length=300)
    bpm: int | None = Field(default=None, ge=20, le=300)
    key: str = Field(default="", max_length=80)
    scale: str = Field(default="", max_length=80)
    mood: str = Field(default="", max_length=2000)
    scenario: str = Field(default="", max_length=2000)
    production: str = Field(default="", max_length=2000)
    vocal_gender: str = Field(default="", max_length=300)
    timbre: str = Field(default="", max_length=2000)
    performance: str = Field(default="", max_length=2000)
    harmonies: str = Field(default="", max_length=2000)
    vocal_effects: str = Field(default="", max_length=2000)
    instruments: str = Field(default="", max_length=2000)
    groove: str = Field(default="", max_length=2000)
    bass: str = Field(default="", max_length=2000)
    percussion: str = Field(default="", max_length=2000)
    textures: str = Field(default="", max_length=2000)
    spatial_effects: str = Field(default="", max_length=2000)
    instrumental: bool = False
    lyrics: str = Field(default="", max_length=30000)
    sections: list[Section] = Field(default_factory=list, max_length=60)
    duration: float = Field(default=60, ge=5, le=300, description="Maximum seconds; the model can finish earlier")
    seed: int = Field(default=0, ge=0, le=MAX_SEED, description="Seed used when randomization is off (--fixed-seed)")
    randomize_seed: bool = True
    headroom: float = Field(default=0, ge=0, le=60, description="Seconds of decoded audio to buffer before live preview")
    steps: int = Field(default=30, ge=4, le=60, description="Flow-matching steps per window; 30 recommended for listening. Very low values such as 4 are for speed tests and can severely degrade audio")
    guidance: float = Field(default=1.7, ge=1, le=4)
    offload: Literal["model", "none", "sequential"] = "model"

    def caption(self) -> str:
        def group(title, extra, pairs):
            values = [f"{label}: {value}." for label, value in pairs if value not in ("", None)]
            if extra.strip():
                values.append(extra.strip())
            return f"{title}\n" + " ".join(values) if values else ""

        global_text = group("Global Metadata", self.global_metadata, [
            ("Genre", self.genre), ("Subgenre", self.subgenre), ("BPM", self.bpm),
            ("Key", self.key), ("Scale", self.scale), ("Emotional progression", self.mood),
            ("Listening scenario", self.scenario), ("Production", self.production)])
        vocals = "Vocal Details\nInstrumental; no vocals." if self.instrumental else group(
            "Vocal Details", self.vocal_details, [("Vocal gender", self.vocal_gender),
            ("Timbre", self.timbre), ("Performance", self.performance),
            ("Harmony and backing vocals", self.harmonies), ("Vocal effects", self.vocal_effects)])
        arrangement = group("Arrangement", self.arrangement, [("Instruments", self.instruments),
            ("Groove", self.groove), ("Bass", self.bass), ("Percussion", self.percussion),
            ("Textures", self.textures), ("Spatial effects", self.spatial_effects)])
        directions = [f"[{s.name}] {s.direction.strip()}" for s in self.sections if s.direction.strip()]
        if directions:
            arrangement = (arrangement or "Arrangement") + "\n" + "\n".join(directions)
        return "\n\n".join(v for v in [self.prompt.strip(), global_text, vocals, arrangement] if v)

    def rendered_lyrics(self) -> str:
        if self.instrumental:
            return "[instrumental]"
        if self.sections:
            return "\n\n".join(f"[{s.name}]\n{s.lyrics.strip()}".strip() for s in self.sections)
        return self.lyrics.strip()

    @model_validator(mode="after")
    def validate_song(self):
        if self.sections and self.lyrics.strip():
            raise ValueError("Use section chart lyrics or raw lyrics, not both")
        if not self.caption().strip():
            raise ValueError("Provide a music description or structured caption")
        lyrics = self.rendered_lyrics()
        if not lyrics:
            raise ValueError("Provide lyrics, a section chart, or select Instrumental")
        for line in lyrics.splitlines():
            if re.match(r"^\s*\[[^\]]+\].*\S", line) and re.sub(r"\[[^\]]+\]", "", line).strip():
                raise ValueError("Put section tags on their own lines; the model would discard words on a tag line")
        return self

    def resolved(self):
        return self.model_copy(update={"seed": secrets.randbelow(MAX_SEED + 1) if self.randomize_seed else self.seed,
                                       "randomize_seed": False})


PRESETS = {
    "daybreak": dict(title="Daybreak", genre="Acoustic pop", bpm=96, key="C", scale="major",
        mood="Intimate verses opening into a hopeful chorus", vocal_gender="Female", timbre="Warm and breathy",
        harmonies="Light stacked harmonies in the chorus", instruments="Fingerpicked acoustic guitar, piano, upright bass",
        production="Natural room sound, warm and spacious",
        sections=[dict(name="verse", lyrics="Morning spills across the floor\nI leave my worries by the door", direction="Solo guitar, close vocal, very soft piano"),
                  dict(name="chorus", lyrics="Let the daylight carry me\nInto all that I can be", direction="Brushed drums and upright bass enter; open stereo harmonies")]),
    "after-hours": dict(title="After hours", genre="Lo-fi jazz", bpm=78, key="D", scale="minor", instrumental=True,
        instruments="Rhodes piano, muted trumpet, upright bass", groove="Lazy swung pocket", percussion="Soft brushes",
        arrangement="[intro] Vinyl texture and Rhodes. [verse] Bass and brushes enter. [chorus] Muted trumpet melody. [outro] Rhodes fades alone.",
        production="Warm tape saturation and an intimate jazz room"),
    "neon": dict(title="Neon horizon", genre="Synth pop", bpm=118, key="A", scale="minor", vocal_gender="Male",
        timbre="Clear, low and expressive", instruments="Analog synths, drum machine, pulsing bass", mood="Reflective to euphoric",
        sections=[dict(name="verse", lyrics="Streetlights draw a silver line\nThrough the quiet of my mind", direction="Filtered arpeggio and dry vocal"),
                  dict(name="chorus", lyrics="We are sparks against the night\nWe will turn the dark to light", direction="Wide synth chords, full drums, octave backing vocals")]),
}
