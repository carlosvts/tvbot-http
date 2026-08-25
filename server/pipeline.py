#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — Máquina de estados de uma sessão da TV Box.

    IDLE ──(wake word)──► RECORDING ──(VAD/teto)──► BUSY ──(áudio tocado)──► IDLE
      ▲                       │                                              │
      └───────(abortado)──────┘                                              │
      └──────────────────────────────────────────────────────────────────────┘

Durante BUSY o áudio que chega é **descartado**. Isso resolve dois problemas de
uma vez: o microfone não fica acumulando enquanto o Whisper/Portal rodam, e a
própria resposta do TTS saindo pelo Anker não volta pelo microfone e dispara a
wake word de novo. Por isso o estado só volta a IDLE depois da duração do WAV
mais uma margem (PLAYBACK_GUARD_S).
"""

import asyncio
import logging
import time
from enum import Enum, auto

import portal
import stt
import tts
from config import cfg
from protocol import OP_AUDIO, OP_BEEP, pack_frame
from vad import SegmentResult, UtteranceSegmenter
from wakeword import WakeWordDetector

log = logging.getLogger(__name__)


class State(Enum):
    IDLE = auto()          # esperando a wake word
    RECORDING = auto()     # capturando a frase
    BUSY = auto()          # processando / tocando a resposta


class Session:
    """Uma conexão da TV Box. Consome PCM e devolve frames pelo socket."""

    def __init__(self, writer: asyncio.StreamWriter, peer: str) -> None:
        self.writer = writer
        self.peer = peer
        self.state = State.IDLE
        self.detector = WakeWordDetector()
        self.segmenter = UtteranceSegmenter()
        self._task: asyncio.Task | None = None

    # ─── Entrada de áudio ────────────────────────────────────────────────────

    async def feed(self, chunk: bytes) -> None:
        """Processa um chunk de 30 ms vindo da TV Box."""
        if self.state is State.BUSY:
            return                                  # descarta durante o turno

        if self.state is State.IDLE:
            if self.detector.feed(chunk):
                await self._on_wake_word()
            return

        # RECORDING
        result = self.segmenter.feed(chunk)
        if result is SegmentResult.CONTINUE:
            return

        audio = self.segmenter.audio
        self.segmenter.reset()

        if result is SegmentResult.ABORTED:
            self._back_to_idle()
            return

        self.state = State.BUSY
        self._task = asyncio.create_task(self._handle_utterance(audio))

    # ─── Transições ──────────────────────────────────────────────────────────

    async def _on_wake_word(self) -> None:
        self.state = State.RECORDING
        self.segmenter.reset()
        if cfg.send_beep:
            await self._send(pack_frame(OP_BEEP))
        log.info("🎙 [%s] Gravando a pergunta (teto de %.1fs)...",
                 self.peer, cfg.max_utterance_s)

    def _back_to_idle(self) -> None:
        self.detector.reset()
        self.segmenter.reset()
        self.state = State.IDLE
        log.info("💤 [%s] Aguardando wake word.", self.peer)

    # ─── Turno completo ──────────────────────────────────────────────────────

    async def _handle_utterance(self, pcm: bytes) -> None:
        """STT → Portal → TTS → envia o WAV. Roda fora do loop de leitura."""
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            # 1) STT (bloqueante, GPU) — vai para um thread
            text = await loop.run_in_executor(None, stt.transcribe, pcm)
            if not text:
                log.info("[%s] Transcrição vazia — turno descartado.", self.peer)
                return

            # 2) Portal (RAG + LLM)
            answer = await portal.ask(text)
            if not answer:
                log.warning("[%s] Portal devolveu resposta vazia.", self.peer)
                return

            # 3) TTS (bloqueante, CPU) — também para um thread
            wav = await loop.run_in_executor(None, tts.synthesize, answer)
            if not wav:
                log.warning("[%s] TTS não gerou áudio.", self.peer)
                return

            # 4) Envia para a TV Box
            await self._send(pack_frame(OP_AUDIO, wav))
            duration = tts.wav_duration_s(wav)
            log.info("📤 [%s] Áudio enviado: %.1fs de fala | turno total %.1fs",
                     self.peer, duration, time.monotonic() - t0)

            # 5) Espera terminar de tocar antes de voltar a escutar
            await asyncio.sleep(duration + cfg.playback_guard_s)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("[%s] Falha no turno: %s", self.peer, e, exc_info=True)
        finally:
            self._back_to_idle()

    # ─── Saída ───────────────────────────────────────────────────────────────

    async def _send(self, frame: bytes) -> None:
        if self.writer.is_closing():
            return
        self.writer.write(frame)
        await self.writer.drain()

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
