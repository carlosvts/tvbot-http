#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts.py — Text-to-speech em PT-BR com Piper.

O Piper roda em CPU (ONNX Runtime) e é rápido o bastante para tempo real, o
que deixa a VRAM inteira livre para o Whisper.

Detalhe importante: a resposta do Portal vem em **Markdown** (ele instrui o
Gemini a formatar para o balão do chat). Ler `**negrito**` ou `- item` em voz
alta fica ruim, então há uma limpeza antes de sintetizar.
"""

import io
import logging
import re
import time
import wave

from config import cfg

log = logging.getLogger(__name__)

_voice = None


def load_voice() -> None:
    """Carrega o modelo de voz do Piper. Chamado no startup."""
    global _voice
    if _voice is not None:
        return

    from piper import PiperVoice

    log.info("Carregando voz Piper: %s", cfg.piper_voice)
    t0 = time.monotonic()
    _voice = PiperVoice.load(cfg.piper_voice)
    log.info("✓ Piper pronto em %.1fs", time.monotonic() - t0)


# ─── Limpeza de Markdown ─────────────────────────────────────────────────────

_MD_RULES: list[tuple[str, str]] = [
    (r"```.*?```", " "),                    # blocos de código
    (r"`([^`]*)`", r"\1"),                  # código inline
    (r"!\[[^\]]*\]\([^)]*\)", " "),         # imagens
    (r"\[([^\]]+)\]\([^)]*\)", r"\1"),      # links → só o texto
    (r"^\s{0,3}#{1,6}\s*", " "),          # headers
    (r"\*\*([^*]+)\*\*", r"\1"),            # negrito
    (r"\*([^*]+)\*", r"\1"),                # itálico
    (r"__([^_]+)__", r"\1"),
    (r"^\s*[-*+]\s+", " "),               # bullets
    (r"^\s*>\s?", " "),                   # citações
    (r"\|", " "),                           # tabelas
    (r"^\s*[-:| ]{3,}\s*$", " "),           # separadores de tabela
]


def clean_for_speech(text: str) -> str:
    """Tira marcação de Markdown e emojis para a fala sair natural."""
    out = text
    for pattern, repl in _MD_RULES:
        out = re.sub(pattern, repl, out, flags=re.MULTILINE | re.DOTALL)

    # Emojis e símbolos fora do BMP latino
    out = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", " ", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{2,}", ". ", out)
    out = out.replace("\n", ". ")
    out = re.sub(r"\.{2,}", ".", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = out.strip()

    if cfg.tts_max_chars > 0 and len(out) > cfg.tts_max_chars:
        cut = out[: cfg.tts_max_chars]
        # corta na última fronteira de frase para não terminar no meio da palavra
        last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        out = cut[: last + 1] if last > cfg.tts_max_chars // 2 else cut + "..."
        log.info("Resposta truncada para %d chars antes do TTS.", len(out))

    return out


# ─── Síntese ─────────────────────────────────────────────────────────────────

def synthesize(text: str) -> bytes:
    """
    Sintetiza o texto e devolve um arquivo WAV completo em bytes.
    Bloqueante — chamar de dentro de um executor.
    """
    if _voice is None:
        raise RuntimeError("Voz do Piper não carregada. Chame load_voice() primeiro.")

    speech = clean_for_speech(text)
    if not speech:
        return b""

    t0 = time.monotonic()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        # A API do Piper mudou de nome entre versões; cobrimos as duas.
        if hasattr(_voice, "synthesize_wav"):
            _voice.synthesize_wav(speech, wf)
        else:
            _voice.synthesize(speech, wf)

    data = buf.getvalue()
    log.info("TTS em %.2fs (%d chars → %d KB)",
             time.monotonic() - t0, len(speech), len(data) // 1024)
    return data


def wav_duration_s(wav_bytes: bytes) -> float:
    """Duração do WAV em segundos — usada para saber quando voltar a escutar."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0
