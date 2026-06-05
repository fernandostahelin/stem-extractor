# stem-extractor

Separa **stems** de uma música usando **dois modelos independentes** e salva as
saídas lado a lado para comparação:

| Modelo | Como roda | Stems |
|---|---|---|
| **Demucs** (`htdemucs_ft`) | PyTorch, acelerado por **MPS** no Apple Silicon | 4 stems (bass SDR ~12.0) |
| **BS-RoFormer-SW** | via [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) | 6 stems SOTA: bass/drums/vocals/guitar/piano/other |

Sempre gera também `no_bass.wav` — a música **sem o contrabaixo**.

> **Limite conhecido (baixo × bumbo):** quando o contrabaixo e o kick ocupam
> as mesmas frequências graves, nenhum modelo separa 100%. Medido aqui, tanto
> o Demucs quanto o BS-RoFormer-SW atribuem ~45% da energia de 60–250 Hz ao
> baixo e ~43% à bateria — então o `no_bass` ainda pode ter grave audível.

## Requisitos

- [`uv`](https://docs.astral.sh/uv/) e `ffmpeg` no PATH.
- O PyTorch ainda não publica wheels para Python 3.14, então o projeto fixa
  **Python 3.12** (`.python-version`); o `uv` baixa essa versão automaticamente.

## Setup

```bash
uv sync          # cria o venv (Python 3.12) e instala as dependências
```

## Uso

```bash
# 1) Fase de teste — extrai SÓ o baixo dos dois modelos (rápido)
uv run stem_extractor.py "examples/minha musica.mp3"

# Teste ainda mais rápido: só os primeiros 30s
uv run stem_extractor.py "examples/minha musica.mp3" --duration 30

# 2) Separação completa dos dois modelos
uv run stem_extractor.py "examples/minha musica.mp3" --mode full

# Rodar só um modelo / trocar device
uv run stem_extractor.py "examples/minha musica.mp3" --only demucs --device cpu

# Listar checkpoints RoFormer disponíveis
uv run stem_extractor.py --list-models
```

> O **primeiro run baixa os pesos** dos modelos (algumas centenas de MB); os
> seguintes usam o cache.

## Saída

```
output/<musica>/
  demucs/       bass.wav, no_bass.wav   (+ drums/vocals/other no modo full)
  bs_roformer/  bass.wav, no_bass.wav   (+ drums/vocals/guitar/piano/other no modo full)
```

## Notas

- O segundo modelo fica na constante `SECOND_MODEL` em `stem_extractor.py`.
  Use `--list-models` para ver alternativas e troque ali.
- Áudio (`*.mp3`, `*.wav`, ...) e a pasta `output/` são **gitignored** —
  exemplos comerciais e stems gerados ficam só no disco local.
