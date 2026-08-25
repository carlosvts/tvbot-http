#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  main.py — O CÉREBRO (roda no PC, DENTRO do Docker)                         ║
║                                                                              ║
║  Recebe áudio da TV Box por TCP e executa o fluxo:                          ║
║                                                                              ║
║      wake word (openWakeWord)                                                ║
║        → grava a frase (webrtcvad + teto de tempo)                          ║
║        → STT (faster-whisper, CUDA)                                          ║
║        → POST /api/chat do PortalTCC (RAG + Gemini)                         ║
║        → TTS (Piper, PT-BR, CPU)                                             ║
║        → devolve o WAV para a TV Box tocar no Anker                         ║
║                                                                              ║
║  Não fala com o Go2 nem com nada além disso.                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import signal
import sys

import portal
import stt
import tts
from config import cfg
from pipeline import Session

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s [CEREBRO] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> None:
    """Uma conexão da TV Box: lê chunks de tamanho fixo e alimenta a sessão."""
    addr = writer.get_extra_info("peername")
    peer = f"{addr[0]}:{addr[1]}" if addr else "?"
    log.info("✓ TV Box conectada: %s", peer)

    session = Session(writer, peer)
    try:
        while True:
            # readexactly garante o alinhamento dos frames de 30 ms, que o VAD
            # e o openWakeWord exigem.
            chunk = await reader.readexactly(cfg.chunk_bytes)
            await session.feed(chunk)
    except asyncio.IncompleteReadError:
        log.info("TV Box %s encerrou o stream.", peer)
    except (ConnectionResetError, BrokenPipeError):
        log.warning("Conexão com %s caiu.", peer)
    except Exception as e:
        log.error("Erro na sessão %s: %s", peer, e, exc_info=True)
    finally:
        await session.close()
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        log.info("Sessão %s finalizada.", peer)


async def main() -> None:
    log.info("=" * 70)
    log.info("  CEREBRO — TVBot + Portal Cafeicultura")
    log.info("  Escuta        : %s:%d", cfg.bind_host, cfg.tcp_port)
    log.info("  Áudio         : %dHz | chunk %dms (%d bytes)",
             cfg.sample_rate, cfg.chunk_ms, cfg.chunk_bytes)
    log.info("  Wake word     : %s (threshold %.2f)", cfg.oww_model, cfg.oww_threshold)
    log.info("  Captura frase : VAD (silêncio %dms) | teto %.1fs",
             cfg.vad_silence_ms, cfg.max_utterance_s)
    log.info("  STT           : faster-whisper %s @ %s",
             cfg.whisper_model, cfg.whisper_device)
    log.info("  TTS           : Piper %s", cfg.piper_voice)
    log.info("  Portal        : %s", cfg.portal_api_url)
    log.info("=" * 70)

    # Carrega os modelos pesados ANTES de aceitar conexões, para a primeira
    # pergunta não pagar o custo de warm-up.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stt.load_model)
    await loop.run_in_executor(None, tts.load_voice)
    await portal.startup()

    server = await asyncio.start_server(handle_client, cfg.bind_host, cfg.tcp_port)

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    log.info("🟢 Pronto. Aguardando a TV Box conectar...")
    async with server:
        await stop.wait()

    log.info("Encerrando...")
    await portal.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
