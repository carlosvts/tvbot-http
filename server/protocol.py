#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol.py — Protocolo do canal reverso (server → TV Box).

O canal de ida (TV Box → server) continua sendo um fluxo cru de PCM:
    int16 little-endian, mono, 16000 Hz, em chunks de 30 ms (960 bytes).

O canal de volta agora precisa carregar áudio, e não só um byte de beep.
Formato de cada frame:

    ┌────────┬──────────────┬──────────────────┐
    │ opcode │  length      │     payload      │
    │ 1 byte │ 4 bytes (BE) │  `length` bytes  │
    └────────┴──────────────┴──────────────────┘

Opcodes:
    0x01 BEEP   → length = 0. A TV Box toca o beep local (confirma wake word).
    0x02 AUDIO  → payload = arquivo WAV completo, tocado pelo Anker.

Manter o header de tamanho fixo evita qualquer ambiguidade de framing sobre
TCP, que é um stream sem fronteiras de mensagem.
"""

import struct

OP_BEEP: int = 0x01
OP_AUDIO: int = 0x02

HEADER_FORMAT = ">BI"          # opcode (uint8) + length (uint32 big-endian)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)   # 5 bytes

# Limite de sanidade para o payload (evita alocar memória absurda se o stream
# dessincronizar por qualquer motivo).
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024   # 32 MB


def pack_frame(opcode: int, payload: bytes = b"") -> bytes:
    """Monta um frame completo pronto para ser escrito no socket."""
    return struct.pack(HEADER_FORMAT, opcode, len(payload)) + payload


def unpack_header(header: bytes) -> tuple[int, int]:
    """Lê um header de HEADER_SIZE bytes e devolve (opcode, length)."""
    return struct.unpack(HEADER_FORMAT, header)
