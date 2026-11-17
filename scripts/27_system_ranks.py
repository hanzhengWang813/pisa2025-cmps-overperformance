"""
27_system_ranks.py — 体系层残差的"分数点"口径与排名

为什么单独一个脚本
  11_residual.py 把残差在体系内做了 z 标准化，写进 resid.csv.gz 的是 resid1–resid10
  与 relperf（z 值），原始残差（CMPS 分制）当场丢弃。于是论文里"低于预测 28.0 分"
  "83 个体系中排第 77"这类表述，无法从已有输出复原；12_descriptive.py 的体系表也只
  按 CMPS 排序，没有残差排名。本脚本补上这一段，使这两个数字可追溯。

口径（与正文一致）
  1. 对 10 组似真值分别做国际池化三次多项式回归（与 11_residual.py 完全相同）
  2. 取原始残差（CMPS 分制，未标准化）
  3. 体系内按学生权重求均值，再对 10 组取平均 → 体系残差（分）
  4. 排名按残差降序：第 1 名 = 最"超出预测"
  5. 同时给出 10 组似真值平均的 CMPS 与数学分，以及 84 个体系的绝对排名
     （表 1 用的是 10 组均值；12_descriptive.py 的 tab1_country.csv 用 PV1，两者
      在 QCI 上差约 0.4 分，属口径差异，非错误）

产出 output/tab_system_resid.csv
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

N_PV = 10


def fit_residual(d: pd.DataFrame, y: str, ms: list[str]) -> np.ndarray:
    """三次多项式回归取残差（与 11_residual.py 同式）。"""
    parts = [np.ones((len(d), 1))]
    for m in ms:
        v = d[m].to_numpy(dtype=float)
        z = (v - v.mean()) / (v.std() or 1.0)
        parts.append(np.column_stack([z, z ** 2, z ** 3]))
    X = np.hstack(parts)
    yv = d[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return yv - X @ beta


def wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna() & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.sum() else np.nan


def main() -> None:
    cols = ({"CNT", "w"}
            | {f"pv{i}_{s}" for i in range(1, N_PV + 1) for s in ("cmps", "math", "read", "scie")})
    df = pd.read_csv(CLEAN / "analysis.csv.gz", usecols=lambda c: c in cols)
    print(f"读入 {len(df):,} 行")

    resid, cmps, math = {}, {}, {}
    for i in range(1, N_PV + 1):
        # 绝对分：所有有 CMPS 的学生都算，与表 1 口径一致（含没有学科成绩的体系）
        for cnt, g in df.groupby("CNT"):
            cmps.setdefault(cnt, []).append(wmean(g[f"pv{i}_cmps"], g["w"]))
            math.setdefault(cnt, []).append(wmean(g[f"pv{i}_math"], g["w"]))

        # 残差：只在学科成绩齐全的学生上拟合
        y = f"pv{i}_cmps"
        ms = [f"pv{i}_math", f"pv{i}_read", f"pv{i}_scie"]
        d = df[[y] + ms + ["CNT", "w"]].dropna(subset=[y] + ms)
        d = d.assign(r=fit_residual(d, y, ms))
        for cnt, g in d.groupby("CNT"):
            resid.setdefault(cnt, []).append(wmean(g["r"], g["w"]))

    t = pd.DataFrame({
        "CNT": list(cmps),
        "cmps_10pv": [np.mean(v) for v in cmps.values()],
        "math_10pv": [np.mean(v) for v in math.values()],
    })
    t["resid_pts"] = t["CNT"].map({c: np.mean(v) for c, v in resid.items()})
    # 绝对排名：84 个体系按 CMPS 降序；相对排名：残差可算的体系按残差降序
    t["rank_abs"] = t["cmps_10pv"].rank(ascending=False, method="min").astype("Int64")
    t = t.sort_values("resid_pts", ascending=False, na_position="last").reset_index(drop=True)
    t["rank_resid"] = np.where(t["resid_pts"].notna(), np.arange(1, len(t) + 1), np.nan)
    t = t[["CNT", "cmps_10pv", "math_10pv", "rank_abs", "resid_pts", "rank_resid"]]
    t["rank_resid"] = t["rank_resid"].astype("Int64")
    t.to_csv(OUT / "tab_system_resid.csv", index=False)

    n_abs = int(t["rank_abs"].notna().sum())
    n_res = int(t["resid_pts"].notna().sum())
    print(f"绝对排名体系数 {n_abs}；残差可算体系数 {n_res}")
    q = t[t["CNT"] == "QCI"].iloc[0]
    print(f"\nQCI：CMPS {q.cmps_10pv:.1f}（第 {q.rank_abs:.0f} / {n_abs}），"
          f"数学 {q.math_10pv:.1f}，残差 {q.resid_pts:+.1f} 分，"
          f"相对排名 {int(q.rank_resid)} / {n_res}")
    top = t.head(3)
    print("残差最高的三个体系：",
          ", ".join(f"{r.CNT} {r.resid_pts:+.1f}" for r in top.itertuples()))
    print(f"已写出 {OUT / 'tab_system_resid.csv'}")


if __name__ == "__main__":
    main()
