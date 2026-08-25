# cerebro-cafe

Server de voz para o [PortalTCC](https://github.com/daviresende123/PortalTCC).
Roda dockerizado no PC com a RTX 4060 e concentra **toda** a inteligência do
sistema. A TV Box (repositório `tvbot-audio-client`) é só microfone e
alto-falante.

```
TV Box                          ESTE REPOSITÓRIO (container)              Portal
──────                          ────────────────────────────              ──────
mic 48kHz ─┬─ ÷3 → 16kHz ──TCP──►  openWakeWord (CPU)
           │                            │
           │  ◄──── beep 0x01 ──────────┘
           │                       webrtcvad grava a frase
           │                       (silêncio 800 ms OU teto de 8 s)
           │                            │
           │                       faster-whisper (CUDA) → texto
           │                            │
           │                            └─ POST /api/chat ─────────────► RAG + Gemini
           │                                                                  │
           │                       Piper PT-BR (CPU) ◄──── answer ────────────┘
           │  ◄──── WAV 0x02 ──────────┘
           └─ aplay → Anker
```

Não há nenhuma comunicação com o Go2. O container faz só o fluxo acima.

---

## Arquivos

```
cerebro-cafe/
├── docker-compose.yml        porta, volume de modelos, acesso à GPU
├── Dockerfile                CUDA 12.4 + cuDNN 9 (exigência do CTranslate2)
├── requirements.txt
├── .env.example              copie para .env
├── models/                   volume: voz do Piper, cache do Whisper, wake word custom
└── server/
    ├── main.py               TCP server asyncio; carrega os modelos no startup
    ├── config.py             toda a configuração, lida do .env
    ├── protocol.py           framing do canal reverso (opcode + tamanho + payload)
    ├── pipeline.py           máquina de estados IDLE → RECORDING → BUSY
    ├── wakeword.py           openWakeWord
    ├── vad.py                webrtcvad com teto de tempo
    ├── stt.py                faster-whisper
    ├── portal.py             cliente do POST /api/chat
    ├── tts.py                Piper + limpeza de Markdown
    └── download_models.py    baixa wake word e voz no primeiro start
```

---

## Subindo

Pré-requisitos: Docker, `nvidia-container-toolkit` e o PortalTCC já rodando.

```bash
cp .env.example .env
nano .env          # PORTAL_API_URL e OWW_MODEL
docker compose up --build
```

O primeiro start baixa o modelo do Whisper (~500 MB no `small`), a voz do Piper
(~60 MB) e a wake word (~2 MB). Tudo cai em `./models` e não é baixado de novo.
Quando aparecer `🟢 Pronto. Aguardando a TV Box conectar...`, está no ar — aí é
só subir o cliente na TV Box.

### Onde está o Portal?

| Situação | `PORTAL_API_URL` |
|---|---|
| Portal rodando no host | `http://host.docker.internal:8000/api/chat` (padrão) |
| Portal em outro compose | `http://backend:8000/api/chat` + rede externa |

Para o segundo caso, o `docker-compose.yml` tem a seção de rede externa
comentada no final — descomente e confira o nome em `docker network ls`.

---

## Decisões que valem conhecer

**Wake word roda aqui, não na TV Box.** A TV Box tem 2 GB de RAM e manda
streaming contínuo; é o `openWakeWord` deste container que decide quando
começar a escutar. Custa pouca CPU e mantém o cliente trivial.

**VAD com teto de tempo.** O `webrtcvad` encerra a frase no silêncio, mas
`MAX_UTTERANCE_S` corta de qualquer jeito. Em ambiente barulhento o silêncio
pode nunca acontecer, e sem o teto o turno travaria para sempre.

**O áudio é descartado durante o estado BUSY.** Enquanto Whisper, Portal e
Piper trabalham — e enquanto a resposta está tocando — os chunks que chegam são
jogados fora. Isso evita acumular minutos de microfone numa fila e impede que a
resposta saindo pelo Anker volte pelo microfone e dispare a wake word de novo.
O estado só volta a `IDLE` depois da duração do WAV mais `PLAYBACK_GUARD_S`.

**`session_id` fixo por processo.** O Portal guarda as últimas 10 trocas, então
perguntas de acompanhamento (*"e a média dela?"*) funcionam. Para zerar,
reinicie o container ou fixe outro `PORTAL_SESSION_ID`.

**Endpoint `/api/chat`, não `/api/chat/stream`.** O TTS precisa do texto
inteiro antes de sintetizar, então o SSE só traria complexidade.

**Markdown é limpo antes do TTS.** O Portal instrui o Gemini a formatar para o
balão do chat; ler `**negrito**` ou `- item` em voz alta fica péssimo.

**Modelos carregados no startup.** Whisper e Piper sobem antes de o socket
aceitar conexões, para a primeira pergunta não pagar o warm-up.

---

## Ajustes finos

| Sintoma | O que mexer |
|---|---|
| Wake word dispara sozinha | `OWW_THRESHOLD` para 0.6–0.7 |
| Wake word não dispara | `OWW_THRESHOLD` para 0.3–0.4 |
| Corta a frase no meio | `VAD_SILENCE_MS` maior (1200) e `MAX_UTTERANCE_S` maior |
| Demora a fechar a frase | `VAD_SILENCE_MS` menor (500), `VAD_AGGRESSIVENESS=3` |
| Transcrição ruim | `WHISPER_MODEL=medium` |
| STT lento demais | `WHISPER_MODEL=base` ou `WHISPER_COMPUTE_TYPE=int8` |
| Resposta falada longa demais | `TTS_MAX_CHARS` menor |
| Voz muito rápida/lenta | `PIPER_LENGTH_SCALE` (>1 mais lento) |

---

## Wake word customizada

Treine um modelo no [openWakeWord](https://github.com/dscripka/openWakeWord),
coloque o `.onnx` em `./models/wakeword/` e aponte:

```
OWW_MODEL=/models/wakeword/tvbot.onnx
```

O `download_models.py` valida a existência do arquivo e baixa só os modelos
base (melspectrogram + embedding), obrigatórios em qualquer caso.

---

## Protocolo com a TV Box

**Ida:** PCM cru — int16 LE, mono, 16000 Hz, chunks de 30 ms (960 bytes).
**Volta:** `opcode (1B) + tamanho (4B big-endian) + payload`, com
`0x01` = beep (sem payload) e `0x02` = WAV da resposta.

A definição canônica está em `server/protocol.py`. O cliente da TV Box tem as
mesmas 5 constantes copiadas inline — se mudar de um lado, mude do outro.