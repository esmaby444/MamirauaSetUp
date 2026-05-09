from __future__ import annotations

import logging
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, cfg: dict):
        self.device = str(cfg.get("device", "plughw:1,0"))
        self.sample_rate = int(cfg.get("sample_rate", 48000))
        self.channels = int(cfg.get("channels", 1))
        self.bit_depth = str(cfg.get("bit_depth", "S16_LE"))
        self.buffer_time_us = int(cfg.get("buffer_time_us", 500000))
        self.period_time_us = int(cfg.get("period_time_us", 125000))

    def start_recording(self, wav_path: Path, duration_sec: int) -> subprocess.Popen:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "arecord",
            "-D",
            self.device,
            "-q",
            "-f",
            self.bit_depth,
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-d",
            str(duration_sec),
            "-B",
            str(self.buffer_time_us),
            "-F",
            str(self.period_time_us),
            "-t",
            "wav",
            str(wav_path),
        ]
        LOG.info("Starting recording: %s", " ".join(cmd))
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
