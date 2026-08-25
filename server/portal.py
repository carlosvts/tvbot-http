#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portal.py — Cliente da API do PortalTCC.

Contrato do endpoint (backend/routes/chat.py do PortalTCC):

    POST /api/chat
    body : {"message": "<pergunta>", "session_id": "<uuid opcional>"}
    resp : {"answer": "<markdown>", "session_id": "<uuid>"}

Usamos o endpoint não-streaming: o TTS precisa do texto inteiro de qualquer
jeito, então o SSE só adicionaria complexidade sem ganho de latência real.

O session_id é fixo por processo, então o RAG mantém o histórico da conversa
entre perguntas (o Portal guarda as últimas 10 trocas).
"""

import logging
import time
import uuid

import httpx

from config import cfg

log = logging.getLogger(__name__)

_session_id: str = cfg.portal_session_id or str(uuid.uuid4())
_client: httpx.AsyncClient | None = None


async def startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=cfg.portal_timeout_s)
    log.info("Portal: %s (session_id=%s)", cfg.portal_api_url, _session_id)


async def shutdown() -> None:
    if _client is not None:
        await _client.aclose()


async def ask(question: str) -> str:
    """Manda a pergunta ao Portal e devolve a resposta em texto."""
    global _session_id

    if _client is None:
        raise RuntimeError("Cliente HTTP não inicializado.")

    payload = {"message": question, "session_id": _session_id}

    t0 = time.monotonic()
    resp = await _client.post(cfg.portal_api_url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # O Portal devolve o session_id que usou; adotamos ele para os próximos
    # turnos (cobre o caso de o backend ter reiniciado e criado outro).
    _session_id = data.get("session_id", _session_id)
    answer = (data.get("answer") or "").strip()

    log.info("Portal respondeu em %.2fs (%d chars)", time.monotonic() - t0, len(answer))
    return answer
