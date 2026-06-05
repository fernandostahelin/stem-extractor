#!/usr/bin/env python3
"""Extrai stems de uma música usando dois modelos independentes:

  * Demucs (htdemucs_ft)  — separação híbrida tempo/frequência, acelerada por MPS no M-series.
                            Melhor qualidade de baixo disponível (SDR ~12.0).
  * MDX-Net kuielab bass  — via ``audio-separator``; isolador de baixo dedicado (SDR ~10.4).

Nota: o catálogo do audio-separator não tem nenhum BS-RoFormer que isole o baixo
sozinho (os checkpoints distribuídos são de vocais/instrumental). O kuielab é o
melhor isolador de baixo pronto fora o Demucs — daí a comparação entre os dois.

Modos:
  bass  (padrão)  -> extrai SÓ o baixo de cada modelo (fase de teste / comparação A/B).
  full            -> separação completa do Demucs (4 stems); o kuielab só faz baixo.

Sempre gera também ``no_bass.wav`` (a música sem o contrabaixo) para cada modelo.
As saídas dos dois modelos ficam em pastas separadas, para comparação lado a lado:

  output/<musica>/demucs/    bass.wav, no_bass.wav  (+ drums/vocals/other no modo full)
  output/<musica>/mdx_bass/  bass.wav, no_bass.wav

Uso:
  uv run stem_extractor.py "examples/minha musica.mp3"            # modo bass (teste)
  uv run stem_extractor.py "examples/minha musica.mp3" --mode full
  uv run stem_extractor.py "examples/minha musica.mp3" --duration 30   # só os 30s iniciais
  uv run stem_extractor.py --list-models                          # lista modelos roformer
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DEMUCS_MODEL = "htdemucs_ft"

# Segundo modelo: isolador de baixo dedicado (MDX-Net kuielab) do audio-separator.
# Produz dois stems: "bass" e "no bass". Veja alternativas com --list-models.
SECOND_MODEL = "kuielab_a_bass.onnx"
SECOND_DIR = "mdx_bass"

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"

DEMUCS_STEMS = ("drums", "bass", "other", "vocals")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def die(msg: str) -> None:
    print(f"erro: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str]) -> None:
    print("  →", " ".join(cmd))
    subprocess.run(cmd, check=True)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        die("ffmpeg não encontrado no PATH (necessário para ler/cortar áudio).")


def prepare_input(src: Path, duration: float | None, workdir: Path) -> Path:
    """Devolve o caminho a processar. Com ``duration``, corta os primeiros N
    segundos para um wav temporário (teste rápido no M-series)."""
    if duration is None:
        return src
    trimmed = workdir / f"{src.stem}__first{int(duration)}s.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(duration),
         "-i", str(src), str(trimmed)])
    return trimmed


def sum_stems(stem_files: list[Path], out_path: Path) -> None:
    """Soma vários stems wav numa mix e grava em ``out_path`` (com proteção a clipping)."""
    mix = None
    sr = None
    for f in stem_files:
        data, this_sr = sf.read(f, always_2d=True)
        if mix is None:
            mix, sr = data.astype(np.float64), this_sr
        else:
            n = min(len(mix), len(data))
            mix = mix[:n] + data[:n]
    if mix is None:
        die("nenhum stem para somar.")
    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        mix /= peak
    sf.write(out_path, mix.astype(np.float32), sr)
    print(f"  ✓ {out_path.name} (mix de {len(stem_files)} stems)")


# ---------------------------------------------------------------------------
# Demucs
# ---------------------------------------------------------------------------
def run_demucs(input_path: Path, out_dir: Path, device: str, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp)
        cmd = [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL,
               "-d", device, "-o", str(tmp_out)]
        if mode == "bass":
            cmd += ["--two-stems", "bass"]   # gera bass.wav + no_bass.wav direto
        cmd.append(str(input_path))
        run(cmd)

        produced = tmp_out / DEMUCS_MODEL / input_path.stem
        if mode == "bass":
            shutil.copy(produced / "bass.wav", out_dir / "bass.wav")
            shutil.copy(produced / "no_bass.wav", out_dir / "no_bass.wav")
            print(f"  ✓ bass.wav, no_bass.wav")
        else:
            for stem in DEMUCS_STEMS:
                shutil.copy(produced / f"{stem}.wav", out_dir / f"{stem}.wav")
            sum_stems([out_dir / f"{s}.wav" for s in DEMUCS_STEMS if s != "bass"],
                      out_dir / "no_bass.wav")
    print(f"  ✓ Demucs → {out_dir}")


# ---------------------------------------------------------------------------
# Segundo modelo: MDX-Net kuielab bass (via audio-separator)
# ---------------------------------------------------------------------------
def _label(path: Path) -> str:
    """Extrai o rótulo do stem do nome de arquivo do audio-separator,
    ex.: 'track_(Bass)_model.wav' -> 'bass'.

    Usa o ÚLTIMO grupo entre parênteses: o nome da faixa pode conter os seus
    próprios parênteses (ex.: '(Official Music Video)'), e a tag do stem é
    sempre a última que o audio-separator acrescenta."""
    groups = re.findall(r"\(([^)]+)\)", path.name)
    return (groups[-1] if groups else path.stem).strip().lower()


def run_second_model(input_path: Path, out_dir: Path, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp)
        run(["audio-separator", str(input_path),
             "--model_filename", SECOND_MODEL,
             "--output_dir", str(tmp_out),
             "--output_format", "WAV"])
        labeled = [(_label(p), p) for p in sorted(tmp_out.glob("*.wav"))]
        if not labeled:
            die("audio-separator não gerou nenhum arquivo.")

        bass = next((p for lbl, p in labeled if lbl == "bass"), None)
        if bass is None:
            die(f"sem stem de baixo. Saídas: {[lbl for lbl, _ in labeled]}")
        shutil.copy(bass, out_dir / "bass.wav")

        no_bass = next((p for lbl, p in labeled
                        if lbl in ("no bass", "no_bass", "instrumental")), None)
        if no_bass is not None:
            shutil.copy(no_bass, out_dir / "no_bass.wav")
        else:
            others = [p for lbl, p in labeled if lbl != "bass"]
            if others:
                sum_stems(others, out_dir / "no_bass.wav")

        if mode == "full":
            for lbl, p in labeled:
                if lbl == "bass" or "no bass" in lbl or lbl == "instrumental":
                    continue
                shutil.copy(p, out_dir / f"{lbl.replace(' ', '_')}.wav")
    print(f"  ✓ {SECOND_MODEL} → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Separa stems com Demucs e MDX-Net kuielab bass (saídas lado a lado).")
    ap.add_argument("input", nargs="?", type=Path,
                    help="arquivo de áudio (wav/mp3/flac/...)")
    ap.add_argument("--mode", choices=["bass", "full"], default="bass",
                    help="bass = só o baixo (padrão); full = todos os stems")
    ap.add_argument("--device", default="mps",
                    help="device do Demucs: mps (padrão) | cpu | cuda")
    ap.add_argument("--duration", type=float, default=None,
                    help="processa só os primeiros N segundos (teste rápido)")
    ap.add_argument("--only", choices=["demucs", "mdx"], default=None,
                    help="rodar apenas um dos modelos")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"pasta de saída base (padrão: {OUTPUT_ROOT})")
    ap.add_argument("--list-models", action="store_true",
                    help="lista os modelos do audio-separator e sai")
    args = ap.parse_args()

    if args.list_models:
        subprocess.run(["audio-separator", "--list_models"])
        return

    if args.input is None:
        ap.error("informe o arquivo de áudio (ou use --list-models).")
    if not args.input.exists():
        die(f"arquivo não encontrado: {args.input}")
    require_ffmpeg()

    out_base = (args.out or OUTPUT_ROOT) / args.input.stem
    print(f"== {args.input.name}  |  modo: {args.mode}  |  saída: {out_base}")

    with tempfile.TemporaryDirectory() as tmp:
        audio = prepare_input(args.input, args.duration, Path(tmp))

        if args.only in (None, "demucs"):
            print("\n[demucs]")
            run_demucs(audio, out_base / "demucs", args.device, args.mode)
        if args.only in (None, "mdx"):
            print(f"\n[{SECOND_DIR}]")
            run_second_model(audio, out_base / SECOND_DIR, args.mode)

    print(f"\n✓ pronto → {out_base}")


if __name__ == "__main__":
    main()
