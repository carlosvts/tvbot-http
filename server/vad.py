#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vad.py — Segmentação da frase depois da wake word.

Estratégia escolhida: **VAD com teto de tempo**. O webrtcvad decide quando a
frase acabou pelo silêncio, mas existe um limite absoluto (MAX_UTTERANCE_S)
que encerra a gravação de qualquer jeito. Isso é o que impede o fluxo de
travar em ambiente barulhento, onde o silêncio talvez nunca aconteça.

O webrtcvad só aceita frames de 10, 20 ou 30 ms em 8/16/32/48 kHz — o chunk de
30 ms @ 16 kHz que vem da TV Box já está exatamente no formato certo.
"""

import logging
from enum import Enum, auto

import webrtcvad

from config import cfg

log = logging.getLogger(__name__)


class SegmentResult(Enum):
    CONTINUE = auto()      # ainda gravando
    DONE = auto()          # frase capturada com sucesso
    ABORTED = auto()       # ninguém falou / fala curta demais


class UtteranceSegmenter:
    """Acumula chunks e decide quando a frase terminou."""

    def __init__(self) -> None:
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.reset()

    def reset(self) -> None:
        self._frames: list[bytes] = []
        self._elapsed_ms: int = 0
        self._silence_ms: int = 0
        self._speech_ms: int = 0
        self._speech_started: bool = False

    def feed(self, pcm_chunk: bytes) -> SegmentResult:
        """Alimenta um chunk de 30 ms e devolve o estado da segmentação."""
        self._frames.append(pcm_chunk)
        self._elapsed_ms += cfg.chunk_ms

        try:
            is_speech = self.vad.is_speech(pcm_chunk, cfg.sample_rate)
        except Exception:
            # Frame com tamanho inesperado: trata como silêncio em vez de morrer.
            is_speech = False

        if is_speech:
            self._speech_started = True
            self._speech_ms += cfg.chunk_ms
            self._silence_ms = 0
        elif self._speech_started:
            self._silence_ms += cfg.chunk_ms

        # 1) Teto absoluto de duração.
        if self._elapsed_ms >= cfg.max_utterance_s * 1000:
            log.info("⏱ Teto de %.1fs atingido — encerrando captura.", cfg.max_utterance_s)
            return self._finish()

        # 2) Ninguém começou a falar dentro da janela inicial.
        if not self._speech_started and self._elapsed_ms >= cfg.initial_speech_timeout_s * 1000:
            log.info("Nenhuma fala detectada em %.1fs — abortando.",
                     cfg.initial_speech_timeout_s)
            return SegmentResult.ABORTED

        # 3) Silêncio suficiente depois da fala.
        if self._speech_started and self._silence_ms >= cfg.vad_silence_ms:
            log.info("Silêncio de %dms — frase encerrada (%.1fs).",
                     cfg.vad_silence_ms, self._elapsed_ms / 1000)
            return self._finish()

        return SegmentResult.CONTINUE

    def _finish(self) -> SegmentResult:
        if self._speech_ms < cfg.min_speech_ms:
            log.info("Fala curta demais (%dms) — descartando.", self._speech_ms)
            return SegmentResult.ABORTED
        return SegmentResult.DONE

    @property
    def audio(self) -> bytes:
        """PCM int16 16 kHz de tudo que foi gravado."""
        return b"".join(self._frames)
