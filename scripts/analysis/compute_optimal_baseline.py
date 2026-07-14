#!/usr/bin/env python3
"""Baseline ótimo (worst-case): perguntas sim/não que dividem o espaço ao meio.

Assume prior uniforme sobre N candidatos, H = log2(N) (igual a `src.entropy.Entropy`).

Minimax: após cada resposta, o pior caso deixa o maior grupo — tamanho ceil(N/2).
O ganho por turno é H(N) - H(ceil(N/2)); a soma dos IG até N=1 é log2(N).

Usage:
    python scripts/compute_optimal_baseline.py --n 160
    python scripts/compute_optimal_baseline.py --n 160 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.entropy import Entropy


def worst_case_trajectory(n: int) -> list[dict]:
    """Uma partida ótima minimax: n -> ceil(n/2) até restar 1 candidato."""
    rows: list[dict] = []
    turn = 0
    cur = n
    while cur > 1:
        turn += 1
        h_before = Entropy.compute(cur)
        n_after = (cur + 1) // 2  # ceil(cur/2)
        h_after = Entropy.compute(n_after)
        ig = Entropy.info_gain(h_before, h_after)
        rows.append(
            {
                "turn": turn,
                "n_before": cur,
                "n_after": n_after,
                "h_before": h_before,
                "h_after": h_after,
                "info_gain": ig,
            }
        )
        cur = n_after
    return rows


def _fmt_table(rows: list[dict]) -> str:
    lines = [
        f"{'turn':>4}  {'N':>6}  {'N_after':>8}  {'H_before':>10}  {'H_after':>10}  {'IG':>10}",
        "-" * 62,
    ]
    for r in rows:
        lines.append(
            f"{r['turn']:4d}  {r['n_before']:6d}  {r['n_after']:8d}  "
            f"{r['h_before']:10.6f}  {r['h_after']:10.6f}  {r['info_gain']:10.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Baseline IG (worst-case): divisão binária minimax ceil(N/2) por turno."
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Número de hipóteses (candidatos ativos).",
    )
    p.add_argument("--json", action="store_true", help="Saída JSON em vez de texto.")
    args = p.parse_args()

    n = args.n
    if n < 1:
        print("N deve ser >= 1.", file=sys.stderr)
        sys.exit(1)

    h0 = Entropy.compute(n)
    traj = worst_case_trajectory(n)
    total_ig = sum(r["info_gain"] for r in traj)
    n_turns = len(traj)

    out = {
        "n_hypotheses": n,
        "mode": "worst_case_halving",
        "h_initial_bits": h0,
        "turns": n_turns,
        "total_info_gain": total_ig,
        "avg_info_gain_per_turn": total_ig / n_turns if n_turns else 0.0,
        "trajectory": traj,
        "note": "Soma dos IG = log2(N) quando H(1)=0; coincide com resolução do alvo.",
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"N = {n}  |  H inicial = {h0:.6f} bits  |  worst-case (ceil(N/2))\n")
        if traj:
            print(_fmt_table(traj))
        print()
        print(f"Turnos: {n_turns}")
        print(f"IG total:   {total_ig:.6f} bits  (teórico log2(N) = {h0:.6f})")
        print(f"IG médio/turno: {out['avg_info_gain_per_turn']:.6f}")


if __name__ == "__main__":
    main()
