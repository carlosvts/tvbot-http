#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py — Baixa os artefatos que não vêm na imagem.

Roda no entrypoint do container, antes do main.py. É idempotente: se o arquivo
já existe no volume /models, não baixa de novo.

  1. openWakeWord: modelos base (melspectrogram + embedding) e, se OWW_MODEL
     for um nome pré-treinado, o modelo dessa wake word.
  2. Piper: a voz PT-BR indicada em PIPER_VOICE (.onnx + .onnx.json).

Se você usa uma wake word customizada (.onnx montado em /models), o passo 1 só
baixa os modelos base, que são obrigatórios de qualquer jeito.
"""

import os
import sys
import urllib.request
from pathlib import Path

OWW_MODEL = os.getenv("OWW_MODEL", "hey_jarvis")
PIPER_VOICE = os.getenv("PIPER_VOICE", "/models/piper/pt_BR-faber-medium.onnx")

# Vozes PT-BR disponíveis no repositório oficial do Piper no HuggingFace.
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR"
PIPER_PATHS = {
    "pt_BR-faber-medium":   f"{PIPER_BASE}/faber/medium/pt_BR-faber-medium.onnx",
    "pt_BR-edresson-low":   f"{PIPER_BASE}/edresson/low/pt_BR-edresson-low.onnx",
    "pt_BR-cadu-medium":    f"{PIPER_BASE}/cadu/medium/pt_BR-cadu-medium.onnx",
    "pt_BR-jeff-medium":    f"{PIPER_BASE}/jeff/medium/pt_BR-jeff-medium.onnx",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ já existe: {dest}")
        return
    print(f"  ↓ {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  ✓ salvo em {dest} ({dest.stat().st_size // 1024} KB)")


def fetch_wakeword() -> None:
    print("[modelos] openWakeWord")
    import openwakeword.utils as oww_utils

    is_custom = OWW_MODEL.endswith((".onnx", ".tflite")) or os.path.isabs(OWW_MODEL)
    if is_custom:
        if not Path(OWW_MODEL).exists():
            print(f"  ! wake word customizada não encontrada: {OWW_MODEL}", file=sys.stderr)
            print("    monte o arquivo em ./models e ajuste OWW_MODEL no .env", file=sys.stderr)
            sys.exit(1)
        # Os modelos base (melspectrogram + embedding) são obrigatórios mesmo
        # com wake word customizada. Pedir um pré-treinado qualquer os traz
        # junto, sem baixar o catálogo inteiro.
        oww_utils.download_models(model_names=["hey_jarvis"])
        print(f"  ✓ wake word customizada: {OWW_MODEL}")
    else:
        oww_utils.download_models(model_names=[OWW_MODEL])
        print(f"  ✓ wake word pré-treinada: {OWW_MODEL}")


def fetch_piper() -> None:
    print("[modelos] Piper")
    dest = Path(PIPER_VOICE)
    name = dest.stem

    config = Path(str(dest) + ".json")
    if dest.exists() and config.exists():
        print(f"  ✓ já existe: {dest}")
        return

    url = PIPER_PATHS.get(name)
    if url is None:
        print(f"  ! voz desconhecida: {name}", file=sys.stderr)
        print(f"    vozes conhecidas: {', '.join(PIPER_PATHS)}", file=sys.stderr)
        print("    ou coloque o .onnx e o .onnx.json manualmente em ./models/piper/",
              file=sys.stderr)
        sys.exit(1)

    _download(url, dest)
    _download(url + ".json", config)


if __name__ == "__main__":
    fetch_wakeword()
    fetch_piper()
    print("[modelos] tudo pronto.")
