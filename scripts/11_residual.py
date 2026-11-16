"""
11_residual.py — 相对表现（relative overperformance）的构造

复现 OECD (2024) PISA 2022 Volume III 对"相对表现"的定义：
  对学科成绩做三次多项式回归，国际池化，取残差。

本研究的移植：因变量改为 PISA 2025 的 CMPS（计算问题解决）似真值，
预测变量为数学、阅读、科学三科似真值（各取三次多项式项），国际池化拟合。

步骤：
  1. 对 10 组似真值分别拟合，得到 10 组残差
  2. 残差在**体系内**做 z 标准化（使跨体系可比，且消除体系均值差异）
  3. 合并 10 组得 relperf（点估计取均值，方差按 Rubin 保留）
  4. 超额者定义：体系内残差 ≥ 第 75 百分位（敏感性分析用 z ≥ 1）

产出 data/clean/resid.csv.gz（仅含 CNTSTUID 与残差相关列）
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"


def fit_residual(d: pd.DataFrame, y: str, ms: list[str]) -> np.ndarray:
    """对 y 关于 ms 的三次多项式做 OLS，返回残差（与 d 行对齐）。"""
    parts = [np.ones((len(d), 1))]
    for m in ms:
        v = d[m].to_numpy(dtype=float)
        z = (v - v.mean()) / (v.std() or 1.0)
        parts.append(np.column_stack([z, z ** 2, z ** 3]))
    X = np.hstack(parts)
    yv = d[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return yv - X @ beta


def main() -> None:
    df = pd.read_csv(CLEAN / "analysis.csv.gz",
                     usecols=lambda c: c in (
                         {"CNT", "CNTSTUID", "w", "escs", "female", "grade", "has_ldw"}
                         | {f"pv{i}_cmps" for i in range(1, 11)}
                         | {f"pv{i}_math" for i in range(1, 11)}
                         | {f"pv{i}_read" for i in range(1, 11)}
                         | {f"pv{i}_scie" for i in range(1, 11)}))
    print(f"读入 {len(df):,} 行")

    out = df[["CNT", "CNTSTUID", "w", "escs", "female", "grade", "has_ldw"]].copy()

    resid_cols = []
    r2s, ns = [], []
    for i in range(1, 11):
        y, ms = f"pv{i}_cmps", [f"pv{i}_math", f"pv{i}_read", f"pv{i}_scie"]
        d = df[[y] + ms + ["CNT"]].dropna()
        if len(d) < 1000:
            continue
        res = fit_residual(d, y, ms)
        # 拟合优度（供报告：LDW 的方差有多少可由学业基线解释）
        yv = d[y].to_numpy()
        r2 = 1 - res.var() / yv.var()
        r2s.append(r2)
        ns.append(len(d))
        s = pd.Series(np.nan, index=df.index, dtype=float)
        s.loc[d.index] = res
        # 体系内 z 标准化
        g = df["CNT"]
        mu = s.groupby(g).transform("mean")
        sd = s.groupby(g).transform("std")
        col = f"resid{i}"
        out[col] = (s - mu) / sd
        resid_cols.append(col)

    print(f"\n三次多项式回归 R²（10 组似真值）："
          f"均值 {np.mean(r2s):.3f}，范围 {np.min(r2s):.3f}–{np.max(r2s):.3f}")
    print("→ 即 CMPS 方差的约 "
          f"{np.mean(r2s)*100:.0f}% 可由数学/阅读/科学成绩解释，"
          f"剩余 {100-np.mean(r2s)*100:.0f}% 为'学业基线之外'的部分")

    out["relperf"] = out[resid_cols].mean(axis=1)
    out["over_q75"] = np.nan
    out["over_z1"] = np.nan
    for c, sub in out.groupby("CNT"):
        r = sub["relperf"]
        thr = r.quantile(0.75)
        out.loc[sub.index, "over_q75"] = (r >= thr).astype(float)
        out.loc[sub.index, "over_z1"] = (r >= 1.0).astype(float)
    out.loc[out["relperf"].isna(), ["over_q75", "over_z1"]] = np.nan

    print(f"\n超额者比例（全样本，未加权）：q75={out['over_q75'].mean()*100:.1f}%  "
          f"z>=1={out['over_z1'].mean()*100:.1f}%")
    print(f"relperf 有效：{out['relperf'].notna().sum():,}")

    q = out[out["CNT"] == "QCI"]
    print(f"\nQCI（N={len(q):,}）：relperf 均值 {q['relperf'].mean():.3f}，"
          f"超额者(q75) {q['over_q75'].mean()*100:.1f}%")

    p = CLEAN / "resid.csv.gz"
    out[["CNT", "CNTSTUID", "relperf", "over_q75", "over_z1"] + resid_cols].to_csv(
        p, index=False, compression="gzip")
    print(f"\n已写出 {p}")


if __name__ == "__main__":
    main()
