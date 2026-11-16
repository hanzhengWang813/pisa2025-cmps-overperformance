"""
pisa_utils.py — PISA 分析公共函数

提供三件事：
  1. rubin_pool   : 似真值（PV）多重插补的 Rubin 规则合并
  2. residualize  : 三次多项式回归取残差（复现 OECD PISA 2022 创造性思维报告的
                    "relative performance" 定义：对学科成绩做三次多项式回归、
                    国际池化、取残差）
  3. brr_se       : 平衡重复抽样（BRR）标准误，Fay 系数 0.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- Rubin 合并
def rubin_pool(est: np.ndarray, var: np.ndarray, df_complete: int | None = None) -> dict:
    """合并 M 组似真值下的点估计与方差。

    est, var : 长度 M 的数组
    返回：pooled 估计、总方差、标准误、t、双侧 p、自由度
    """
    est = np.asarray(est, dtype=float)
    var = np.asarray(var, dtype=float)
    m = len(est)
    qbar = est.mean()
    ubar = var.mean()
    b = est.var(ddof=1) if m > 1 else 0.0
    tvar = ubar + (1 + 1.0 / m) * b
    se = float(np.sqrt(tvar))

    if m > 1 and b > 0:
        r = (1 + 1.0 / m) * b / ubar if ubar > 0 else np.inf
        df = (m - 1) * (1 + 1.0 / r) ** 2 if np.isfinite(r) else (df_complete or 1e6)
        if df_complete:
            df = min(df, df_complete)
    else:
        df = df_complete or 1e6

    t = qbar / se if se > 0 else np.nan
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), df)) if np.isfinite(t) else np.nan
    return {"est": float(qbar), "se": se, "t": float(t), "p": float(p),
            "df": float(df), "fmi": float((1 + 1.0 / m) * b / tvar) if tvar > 0 else np.nan}


# ------------------------------------------------------- 三次多项式残差化
def cubic_design(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    z = (x - center) / scale
    return np.column_stack([np.ones_like(z), z, z ** 2, z ** 3])


def residualize(df: pd.DataFrame, y: str, xs: list[str],
                group: str | None = None,
                standardize_within: str | None = None) -> pd.Series:
    """对 y 关于 xs 的各三次多项式项做 OLS，取残差。

    group : 若给定，则在该变量各取值内部分别拟合（对应"国际池化"时该列置 None）
    standardize_within : 残差在该变量各取值内做 z 标准化
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)

    blocks = [(None, df)] if group is None else list(df.groupby(group, observed=True))
    for _, sub in blocks:
        need = [y] + xs
        d = sub[need].dropna()
        if len(d) < 50:
            continue
        parts = [np.ones((len(d), 1))]
        for x in xs:
            v = d[x].to_numpy(dtype=float)
            parts.append(cubic_design(v, v.mean(), v.std() or 1.0)[:, 1:])
        X = np.hstack(parts)
        yv = d[y].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        res = yv - X @ beta
        s = pd.Series(res, index=d.index)
        if standardize_within:
            g = df.loc[d.index, standardize_within]
            mu = s.groupby(g).transform("mean")
            sd = s.groupby(g).transform("std")
            s = (s - mu) / sd
        out.loc[d.index] = s
    return out


# ------------------------------------------------------------- BRR 标准误
def brr_se(est_full: float, est_rep: np.ndarray, fay: float = 0.5) -> float:
    """BRR 方差：V = 1/(R(1-f)^2) * sum (est_r - est_full)^2"""
    est_rep = np.asarray(est_rep, dtype=float)
    est_rep = est_rep[np.isfinite(est_rep)]
    r = len(est_rep)
    if r == 0:
        return np.nan
    return float(np.sqrt(np.sum((est_rep - est_full) ** 2) / (r * (1 - fay) ** 2)))


# ------------------------------------------------------------ 描述性辅助
def wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def weighted_quantile(x: pd.Series, w: pd.Series, q: float) -> float:
    m = x.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    xv, wv = x[m].to_numpy(), w[m].to_numpy()
    order = np.argsort(xv)
    xv, wv = xv[order], wv[order]
    cw = np.cumsum(wv) - 0.5 * wv
    return float(np.interp(q * wv.sum(), cw, xv))
