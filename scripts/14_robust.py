"""
14_robust.py — 稳健性检验与体系异质性（RQ4）

检验项：
  R1 残差定义替换：仅控数学（不控阅读、科学）
  R2 剔除快速作答者（rapid_ratio > 0.2），直接回应"猜测"质疑
  R3 剔除中国四省市，看结论是否由单一体系驱动
  R4 超额者定义替换：z ≥ 1
  R5 体系异质性：ESCS 随机斜率模型 + 与体系特征的相关
  R6 Johnson-Neyman：ESCS 斜率随学校数字资源变化的显著区间
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output"


def fit_residual(d, y, ms):
    parts = [np.ones((len(d), 1))]
    for m in ms:
        v = d[m].to_numpy(float)
        z = (v - v.mean()) / (v.std() or 1.0)
        parts.append(np.column_stack([z, z ** 2, z ** 3]))
    X = np.hstack(parts)
    yv = d[y].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return yv - X @ b


def z_within(s: pd.Series, g: pd.Series) -> pd.Series:
    return (s - s.groupby(g).transform("mean")) / s.groupby(g).transform("std")


def main() -> None:
    a = pd.read_csv(CLEAN / "analysis.csv.gz")
    r = pd.read_csv(CLEAN / "resid.csv.gz")[["CNTSTUID", "relperf", "over_q75", "over_z1"]]
    d = a.merge(r, on="CNTSTUID", how="inner")
    d["sch_escs"] = d.groupby("CNTSCHID")["escs"].transform("mean")
    for v in ["digital_shortage", "cs_club", "sw_coverage"]:
        d[f"z_{v}"] = (d[v] - d[v].mean()) / (d[v].std() or 1)
    d["below_city"] = pd.to_numeric(d["below_city"], errors="coerce")

    RHS = ("escs + z_digital_shortage + z_cs_club + z_sw_coverage + below_city + sch_escs"
           " + escs:z_digital_shortage + escs:z_cs_club + escs:below_city")
    VARS = ["escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage", "below_city",
            "sch_escs", "escs:z_digital_shortage", "escs:z_cs_club", "escs:below_city"]

    L = []
    P = L.append
    P("=" * 86)
    P("稳健性检验（因变量：相对表现；加权最小二乘 + 学校聚类稳健标准误）")
    P("=" * 86)

    def run(data, y, rhs, tag, vars_=VARS):
        x = data.dropna(subset=[y] + [t.strip() for t in rhs.replace(":", "+").split("+")]
                        + ["CNTSCHID", "w"])
        m = smf.wls(f"{y} ~ {rhs}", data=x, weights=x["w"]).fit(
            cov_type="cluster", cov_kwds={"groups": x["CNTSCHID"]})
        P(f"\n[{tag}]  N={int(m.nobs):,}  R²={m.rsquared:.4f}")
        out = {"n": int(m.nobs), "r2": float(m.rsquared)}
        for v in vars_:
            b, se, p = float(m.params[v]), float(m.bse[v]), float(m.pvalues[v])
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            P(f"   {v:<32} b={b:+.4f}  SE={se:.4f}  p={p:.4f} {star}")
            out[v] = {"b": b, "se": se, "p": p}
        return out

    res = {}

    # ---------- 基准 ----------
    res["基准"] = run(d, "relperf", RHS, "基准模型（三科控制）")

    # ---------- R1 仅控数学 ----------
    d1 = d.copy()
    pv1 = d1.dropna(subset=["pv1_cmps", "pv1_math"]).copy()
    pv1["res_math"] = np.nan
    res_raw = fit_residual(pv1, "pv1_cmps", ["pv1_math"])
    s = pd.Series(res_raw, index=pv1.index)
    pv1["res_math"] = z_within(s, pv1["CNT"])
    d1 = d1.merge(pv1[["CNTSTUID", "res_math"]], on="CNTSTUID", how="left")
    res["R1 仅控数学"] = run(d1, "res_math", RHS, "R1 残差仅对数学成绩回归")

    # ---------- R2 剔除快速作答者 ----------
    d2 = d[~(d["rapid_ratio"] > 0.2)].copy()
    res["R2 剔除快速作答"] = run(d2, "relperf", RHS, "R2 剔除快速作答率>0.2 者（回应猜测质疑）")

    # ---------- R3 剔除中国 ----------
    d3 = d[d["CNT"] != "QCI"].copy()
    res["R3 剔除中国"] = run(d3, "relperf", RHS, "R3 剔除中国四省市")

    # ---------- R4 z>=1 定义 ----------
    res["R4 z>=1 定义"] = run(d, "over_z1", RHS, "R4 超额者改按 z≥1 定义（线性概率）", VARS)

    # ---------- R5 体系异质性 ----------
    P("\n" + "=" * 86)
    P("R5 体系异质性：ESCS 斜率的体系间差异")
    P("=" * 86)
    sub = d.dropna(subset=["relperf", "escs", "CNT"])
    slopes = []
    for cnt, g in sub.groupby("CNT"):
        if len(g) < 400:
            continue
        m = smf.wls("relperf ~ escs", data=g, weights=g["w"]).fit()
        cm = g["pv1_cmps"].mean()
        mt = g["pv1_math"].mean()
        slopes.append({"CNT": cnt, "n": len(g), "slope": m.params["escs"],
                       "se": m.bse["escs"], "cmps": cm, "math": mt,
                       "gap": cm - mt, "escs_sd": g["escs"].std(),
                       "escs_mean": g["escs"].mean()})
    sl = pd.DataFrame(slopes)
    sl.to_csv(OUT / "tab5_country_slopes.csv", index=False)
    P(f"覆盖 {len(sl)} 个体系（每体系 n≥400）")
    P(f"  ESCS→相对表现 斜率：均值 {sl['slope'].mean():+.4f}，"
      f"SD {sl['slope'].std():.4f}，范围 {sl['slope'].min():+.3f} ~ {sl['slope'].max():+.3f}")
    neg = (sl["slope"] < -1.96 * sl["se"]).sum()
    pos = (sl["slope"] > 1.96 * sl["se"]).sum()
    P(f"  显著为负（补偿方向）：{neg} 个体系；显著为正（优势放大）：{pos} 个体系；"
      f"不显著：{len(sl)-neg-pos} 个")
    P("\n  显著为正（优势放大）的体系：")
    P("   " + ", ".join(sl[sl["slope"] > 1.96 * sl["se"]]["CNT"].tolist()) or "    （无）")
    P("\n  显著为负（补偿方向）的体系：")
    P("   " + ", ".join(sl[sl["slope"] < -1.96 * sl["se"]]["CNT"].tolist()) or "    （无）")
    P("\n  斜率与体系特征的相关：")
    for v, lab in [("gap", "CMPS 与数学之差"), ("cmps", "CMPS 均值"),
                   ("escs_mean", "ESCS 均值"), ("escs_sd", "ESCS 离散度")]:
        c = np.corrcoef(sl["slope"], sl[v])[0, 1]
        P(f"    {lab:<16} r = {c:+.3f}")
        res.setdefault("R5 相关", {})[v] = float(c)
    res["R5 汇总"] = {"mean_slope": float(sl["slope"].mean()),
                     "sd_slope": float(sl["slope"].std()),
                     "n_neg": int(neg), "n_pos": int(pos), "n_sys": int(len(sl))}

    # ---------- R6 Johnson-Neyman ----------
    P("\n" + "=" * 86)
    P("R6 Johnson-Neyman：ESCS 斜率随学校数字资源短缺变化的显著区间")
    P("=" * 86)
    x = d.dropna(subset=["relperf", "escs", "z_digital_shortage", "CNTSTCHID"
                         if "CNTSTCHID" in d else "CNTSCHID", "w"]).copy()
    mjn = smf.wls("relperf ~ escs * z_digital_shortage", data=x, weights=x["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": x["CNTSCHID"]})
    b1, b3 = mjn.params["escs"], mjn.params["escs:z_digital_shortage"]
    v1 = mjn.cov_params().loc["escs", "escs"]
    v3 = mjn.cov_params().loc["escs:z_digital_shortage", "escs:z_digital_shortage"]
    c13 = mjn.cov_params().loc["escs", "escs:z_digital_shortage"]
    P(f"  交互模型：replerf ~ escs × z_digital_shortage")
    P(f"  b(escs) = {b1:+.4f}   b(交互) = {b3:+.4f}")
    P("\n  不同学校数字资源水平下的 ESCS 斜率（z 为学校层标准化值）：")
    for z in [-2, -1, 0, 1, 2]:
        s_ = b1 + b3 * z
        se_ = np.sqrt(v1 + 2 * z * c13 + z ** 2 * v3)
        t_ = s_ / se_
        P(f"    z={z:+d}（{'极度短缺' if z==2 else '严重短缺' if z==1 else '平均水平' if z==0 else '较充足' if z==-1 else '很充足'}）"
          f"  ESCS 斜率 {s_:+.4f}  SE {se_:.4f}  t={t_:+.2f}")
    # 转换点：斜率由负转正的 z 值
    zc = -b1 / b3 if abs(b3) > 1e-9 else np.nan
    zs = x["z_digital_shortage"].to_numpy()
    pct = 100.0 * float((zs <= zc).mean())
    raw = float(x["digital_shortage"].mean() + zc * x["digital_shortage"].std())
    P(f"\n  斜率由负转正的临界点 z = {zc:+.3f}"
      f"（经验百分位 {pct:.1f}%；对应短缺取值约 {raw:.2f}/4）")
    res["R6"] = {"b_escs": float(b1), "b_int": float(b3), "z_cross": float(zc),
                 "pct_cross": pct, "raw_cross": raw}

    txt = "\n".join(L)
    (OUT / "robust_report.txt").write_text(txt, encoding="utf-8")
    (OUT / "robust_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
