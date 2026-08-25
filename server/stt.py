#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stt.py — Speech-to-text com faster-whisper na GPU.

O modelo é carregado uma única vez no startup (singleton). Carregar por
requisição custaria segundos e VRAM à toa.
"""

import logging
import time

import numpy as np

from config import cfg

log = logging.getLogger(__name__)

_model = None


def load_model() -> None:
    """Carrega o modelo Whisper. Chamado no startup do server."""
    global _model
    if _model is not None:
        return

    from faster_whisper import WhisperModel

    log.info(
        "Carregando faster-whisper: modelo=%s device=%s compute=%s",
        cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type,
    )
    t0 = time.monotonic()
    _model = WhisperModel(
        cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )
    log.info("✓ Whisper pronto em %.1fs", time.monotonic() - t0)


def transcribe(pcm: bytes) -> str:
    """
    Transcreve PCM int16 mono 16 kHz e devolve o texto.
    Chamado de dentro de um executor — é bloqueante.
    """
    if _model is None:
        raise RuntimeError("Whisper não foi carregado. Chame load_model() primeiro.")

    # faster-whisper aceita float32 normalizado em [-1, 1].
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    t0 = time.monotonic()
    segments, _info = _model.transcribe(
        audio,
        language=cfg.whisper_language,
        beam_size=cfg.whisper_beam_size,
        vad_filter=False,          # o webrtcvad já cortou a frase
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log.info("STT em %.2fs → %r", time.monotonic() - t0, text)
    return text
