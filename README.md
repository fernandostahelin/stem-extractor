# stem-extractor

Separa **stems** de uma música. Motor padrão: **BS-RoFormer-SW**. O Demucs fica
opcional (`--engine demucs|both`) para comparação em outras faixas.

| Motor | Como roda | Stems |
|---|---|---|
| **BS-RoFormer-SW** (padrão) | via [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) | 6 stems SOTA: bass/drums/vocals/guitar/piano/other |
| **Demucs** (`htdemucs_ft`) | opcional; PyTorch, acelerado por **MPS** no Apple Silicon | 4 stems (bass SDR ~12.0) |

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
# Padrão: BS-RoFormer-SW, modo bass (só o baixo) — rápido
uv run stem_extractor.py "examples/minha musica.mp3"

# Teste ainda mais rápido: só os primeiros 30s
uv run stem_extractor.py "examples/minha musica.mp3" --duration 30

# Separação completa (6 stems)
uv run stem_extractor.py "examples/minha musica.mp3" --mode full

# Também rodar o Demucs para comparar (ou só o Demucs)
uv run stem_extractor.py "examples/minha musica.mp3" --engine both
uv run stem_extractor.py "examples/minha musica.mp3" --engine demucs --device cpu

# Listar modelos disponíveis
uv run stem_extractor.py --list-models
```

> O **primeiro run baixa os pesos** dos modelos (algumas centenas de MB); os
> seguintes usam o cache.

## Saída

```
output/<musica>/
  bs_roformer/  bass.wav, no_bass.wav   (+ drums/vocals/guitar/piano/other no modo full)
  demucs/       bass.wav, no_bass.wav   (só com --engine demucs|both)
```

## Notas

- O modelo do motor RoFormer fica na constante `ROFORMER_MODEL` em
  `stem_extractor.py`. Use `--list-models` para ver alternativas e troque ali.
- Áudio (`*.mp3`, `*.wav`, ...) e a pasta `output/` são **gitignored** —
  exemplos comerciais e stems gerados ficam só no disco local.
