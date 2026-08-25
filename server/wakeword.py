#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wakeword.py — Detecção de wake word com openWakeWord.

O openWakeWord mantém estado interno (buffer de melspectrogram + embeddings),
então cada conexão da TV Box recebe sua própria instância. Alimentamos ele com
os mesmos chunks de 30 ms que chegam pelo socket — ele acumula sozinho.
"""

import logging
import os
import time

import numpy as np

from config import cfg

log = logging.getLogger(__name__)


class WakeWordDetector:
    """Envolve o openWakeWord e aplica threshold + cooldown."""

    def __init__(self) -> None:
        from openwakeword.model import Model

        model_spec = cfg.oww_model
        if os.path.isabs(model_spec) or model_spec.endswith((".onnx", ".tflite")):
            # Modelo customizado montado em /models
            if not os.path.exists(model_spec):
                raise FileNotFoundError(f"Modelo de wake word não encontrado: {model_spec}")
            kwargs = {"wakeword_models": [model_spec]}
            self.label = os.path.splitext(os.path.basename(model_spec))[0]
        else:
            # Modelo pré-treinado, baixado por download_models.py
            kwargs = {"wakeword_models": [model_spec]}
            self.label = model_spec

        self.model = Model(
            inference_framework=cfg.oww_inference_framework,
            **kwargs,
        )
        self._last_trigger: float = 0.0
        log.info(
            "Wake word carregada: %s (threshold=%.2f, framework=%s)",
            self.label, cfg.oww_threshold, cfg.oww_inference_framework,
        )

    def feed(self, pcm_chunk: bytes) -> bool:
        """
        Alimenta um chunk de PCM int16 16 kHz.
        Retorna True se a wake word foi detectada neste chunk.
        """
        samples = np.frombuffer(pcm_chunk, dtype=np.int16)
        scores = self.model.predict(samples)

        best = max(scores.values()) if scores else 0.0
        if best < cfg.oww_threshold:
            return False

        now = time.monotonic()
        if now - self._last_trigger < cfg.oww_cooldown_s:
            return False

        self._last_trigger = now
        log.info("🔑 Wake word detectada (score=%.3f)", best)
        return True

    def reset(self) -> None:
        """
        Zera o buffer interno. Chamado depois de cada interação para que os
        resíduos da frase anterior (e do próprio TTS) não gerem falso positivo.
        """
        try:
            self.model.reset()
        except AttributeError:
            # Versões antigas do openWakeWord não expõem reset().
            for buf in getattr(self.model, "prediction_buffer", {}).values():
                buf.clear()
