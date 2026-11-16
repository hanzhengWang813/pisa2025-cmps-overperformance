"""45_system_fe.py — 体系固定效应检验：合并斜率是否被组间方差稀释。

背景：相对表现在体系内标准化，体系均值恒为 0；ESCS 仍有约 27% 的方差位于体系之间。
不含体系固定效应的合并回归，分子只含组内协变、分母却含组间方差，斜率因此被压向零。
本脚本给出四个估计（加权/未加权 × 合并/体系内），以及含交互的 M4 在体系固定效应下的版本。
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output"


def fit(y, X, w, cluster):
    m = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    return m


def main() -> None:
    r = pd.read_csv(CLEAN / "resid.csv.gz",
                    usecols=["CNTSTUID", "relperf", "over_q75"])
    a = pd.read_csv(CLEAN / "analysis.csv.gz",
                    usecols=["CNTSTUID", "CNT", "CNTSCHID", "escs", "w",
                             "digital_shortage", "cs_club", "sw_coverage", "below_city"])
    d = r.merge(a, on="CNTSTUID", how="inner").dropna(
        subset=["relperf", "escs", "w"]).copy()
    d = d[d["w"] > 0]

    # 体系内加权中心化
    g = d.groupby("CNT")
    xm = g.apply(lambda t: np.average(t["escs"], weights=t["w"]), include_groups=False)
    ym = g.apply(lambda t: np.average(t["relperf"], weights=t["w"]), include_groups=False)
    d["x_c"] = d["escs"] - d["CNT"].map(xm)
    d["y_c"] = d["relperf"] - d["CNT"].map(ym)

    lines = []
    P = lines.append
    P("=" * 78)
    P("体系固定效应检验（因变量：体系内标准化相对表现）")
    P("=" * 78)
    P(f"样本 N = {len(d):,}；体系 {d['CNT'].nunique()}")

    def one(label, y, x, w):
        m = fit(y, sm.add_constant(x), w, d["CNTSCHID"].values)
        P(f"{label:<22s} b = {m.params[1]:+.5f}  SE = {m.bse[1]:.5f}  "
          f"p = {m.pvalues[1]:.4f}  N = {int(m.nobs):,}")
        return m.params[1]

    w = d["w"].values
    y = d["relperf"].values
    x = d["escs"].values
    P("\n[平均梯度]")
    b_pool_w = one("加权 合并", y, x, w)
    b_fe_w = one("加权 体系内(FE)", d["y_c"].values, d["x_c"].values, w)
    b_pool_u = one("未加权 合并", y, x, np.ones_like(w))
    b_fe_u = one("未加权 体系内(FE)", d["y_c"].values, d["x_c"].values, np.ones_like(w))
    P(f"稀释比例（加权合并/FE）= {b_pool_w / b_fe_w:.3f}")

    # 方差分解
    xbar = np.average(x, weights=w)
    vb = np.average((d["CNT"].map(xm).values - xbar) ** 2, weights=w)
    vw = np.average(d["x_c"].values ** 2, weights=w)
    P(f"ESCS 方差：组间 {vb:.3f}，组内 {vw:.3f}，组间占 {100 * vb / (vb + vw):.1f}%")

    # M4 交互（体系固定效应版）
    P("\n[M4 全设定：体系固定效应 vs 合并]")
    dd = d.dropna(subset=["digital_shortage", "cs_club", "sw_coverage", "below_city"]).copy()
    for c in ["digital_shortage", "cs_club", "sw_coverage"]:
        dd["z_" + c] = (dd[c] - dd[c].mean()) / dd[c].std()
    dd["xi"] = dd["escs"] * dd["z_digital_shortage"]
    dd["xc"] = dd["escs"] * dd["z_cs_club"]
    dd["xl"] = dd["escs"] * dd["below_city"]
    cols = ["relperf", "escs", "z_digital_shortage", "z_cs_club", "sw_coverage",
            "below_city", "xi", "xc", "xl"]
    gg = dd.groupby("CNT")
    for v in cols:
        mm = gg.apply(lambda t: np.average(t[v], weights=t["w"]), include_groups=False)
        dd[v + "_c"] = dd[v] - dd["CNT"].map(mm)
    Xc = sm.add_constant(dd[[v + "_c" for v in cols[1:]]])
    mc = sm.WLS(dd["relperf_c"], Xc, weights=dd["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": dd["CNTSCHID"]})
    Xp = sm.add_constant(dd[cols[1:]])
    mp = sm.WLS(dd["relperf"], Xp, weights=dd["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": dd["CNTSCHID"]})
    P(f"{'term':<26s}{'FE b':>10s}{'FE p':>9s}{'pooled b':>11s}{'pooled p':>10s}")
    for k in cols[1:]:
        P(f"{k:<26s}{mc.params[k + '_c']:>+10.4f}{mc.pvalues[k + '_c']:>9.4f}"
          f"{mp.params[k]:>+11.4f}{mp.pvalues[k]:>10.4f}")

    (OUT / "system_fe_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n已写出 output/system_fe_report.txt")


if __name__ == "__main__":
    main()
