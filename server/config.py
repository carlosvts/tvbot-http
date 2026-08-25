#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — Toda a configuração do server, lida de variáveis de ambiente.

Nada aqui tem valor "mágico" hardcoded no meio do código: se um número
importa, ele aparece nesta tela e pode ser mudado pelo .env sem rebuild.
"""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "sim")


@dataclass(frozen=True)
class Config:
    # ─── Rede ────────────────────────────────────────────────────────────────
    bind_host: str = field(default_factory=lambda: _env("BIND_HOST", "0.0.0.0"))
    tcp_port: int = field(default_factory=lambda: _env_int("TCP_PORT", 9876))

    # ─── Áudio (precisa bater com o client_armbian) ──────────────────────────
    sample_rate: int = field(default_factory=lambda: _env_int("AUDIO_SAMPLE_RATE", 16000))
    chunk_ms: int = field(default_factory=lambda: _env_int("AUDIO_CHUNK_MS", 30))

    # ─── Wake word (openWakeWord) ────────────────────────────────────────────
    # OWW_MODEL aceita o nome de um modelo pré-treinado ("hey_jarvis",
    # "alexa", "hey_mycroft"...) OU um caminho absoluto para um .onnx/.tflite
    # customizado montado em /models.
    oww_model: str = field(default_factory=lambda: _env("OWW_MODEL", "hey_jarvis"))
    oww_threshold: float = field(default_factory=lambda: _env_float("OWW_THRESHOLD", 0.5))
    oww_inference_framework: str = field(
        default_factory=lambda: _env("OWW_INFERENCE_FRAMEWORK", "onnx")
    )
    # Tempo mínimo entre dois disparos de wake word, em segundos.
    oww_cooldown_s: float = field(default_factory=lambda: _env_float("OWW_COOLDOWN_S", 2.0))

    # ─── Captura da frase (VAD com teto de tempo) ────────────────────────────
    # webrtcvad: 0 = mais permissivo, 3 = mais agressivo em cortar não-fala.
    vad_aggressiveness: int = field(default_factory=lambda: _env_int("VAD_AGGRESSIVENESS", 2))
    # Silêncio contínuo que encerra a frase.
    vad_silence_ms: int = field(default_factory=lambda: _env_int("VAD_SILENCE_MS", 800))
    # Teto absoluto: corta a gravação mesmo que o VAD nunca veja silêncio.
    # É esse parâmetro que salva o fluxo em ambiente barulhento.
    max_utterance_s: float = field(default_factory=lambda: _env_float("MAX_UTTERANCE_S", 8.0))
    # Tempo máximo esperando a fala começar depois do beep.
    initial_speech_timeout_s: float = field(
        default_factory=lambda: _env_float("INITIAL_SPEECH_TIMEOUT_S", 3.0)
    )
    # Fala mínima acumulada para valer a pena mandar pro STT.
    min_speech_ms: int = field(default_factory=lambda: _env_int("MIN_SPEECH_MS", 300))

    # ─── STT (faster-whisper) ────────────────────────────────────────────────
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = field(
        default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "int8_float16")
    )
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "pt"))
    whisper_beam_size: int = field(default_factory=lambda: _env_int("WHISPER_BEAM_SIZE", 1))

    # ─── Portal (API do site RAG) ────────────────────────────────────────────
    portal_api_url: str = field(
        default_factory=lambda: _env(
            "PORTAL_API_URL", "http://host.docker.internal:8000/api/chat"
        )
    )
    portal_timeout_s: float = field(default_factory=lambda: _env_float("PORTAL_TIMEOUT_S", 90.0))
    # session_id fixo por processo mantém o contexto conversacional do RAG.
    portal_session_id: str = field(default_factory=lambda: _env("PORTAL_SESSION_ID", ""))

    # ─── TTS (Piper) ─────────────────────────────────────────────────────────
    piper_voice: str = field(
        default_factory=lambda: _env("PIPER_VOICE", "/models/piper/pt_BR-faber-medium.onnx")
    )
    piper_length_scale: float = field(
        default_factory=lambda: _env_float("PIPER_LENGTH_SCALE", 1.0)
    )
    # Corta respostas muito longas antes de sintetizar (0 = sem limite).
    tts_max_chars: int = field(default_factory=lambda: _env_int("TTS_MAX_CHARS", 900))

    # ─── Comportamento ───────────────────────────────────────────────────────
    # Toca o beep na TV Box ao detectar a wake word.
    send_beep: bool = field(default_factory=lambda: _env_bool("SEND_BEEP", True))
    # Margem extra (s) depois da duração do WAV antes de voltar a escutar.
    playback_guard_s: float = field(default_factory=lambda: _env_float("PLAYBACK_GUARD_S", 0.8))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ─── Derivados ───────────────────────────────────────────────────────────
    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000        # 480

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * 2                          # 960


cfg = Config()
