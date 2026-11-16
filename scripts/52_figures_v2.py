# -*- coding: utf-8 -*-
"""52_figures_v2.py — 重绘五张图：去标题、统一口径、按全模型分位置"""
from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

NAVY, RED, GREY, BLUE = "#1f3b73", "#c0392b", "#7f8c8d", "#2e86c1"


def load():
    a = pd.read_csv(CLEAN / "analysis.csv.gz")
    r = pd.read_csv(CLEAN / "resid.csv.gz")[["CNTSTUID", "relperf", "over_q75"]]
    d = a.merge(r, on="CNTSTUID", how="inner")
    for v in ["digital_shortage", "cs_club", "sw_coverage"]:
        d[f"z_{v}"] = (d[v] - d[v].mean()) / (d[v].std() or 1)
    d["below_city"] = pd.to_numeric(d["below_city"], errors="coerce")
    return d


def fig1() -> None:
    """与表 1 同口径：加权、10 组似真值平均。"""
    t = pd.read_csv(ROOT / "output" / "tab1_country.csv").query("n >= 400")
    g = t.dropna(subset=["cmps", "math"])
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.scatter(g["math"], g["cmps"], s=16, c=GREY, alpha=.65, edgecolor="none")
    b1, b0 = np.polyfit(g["math"], g["cmps"], 1)
    xx = np.linspace(g["math"].min() - 8, g["math"].max() + 8, 60)
    ax.plot(xx, b0 + b1 * xx, ls="-", lw=.9, c=GREY)
    lab = {"QCI": (-10, 10), "MYS": (7, -6), "BRN": (7, 2), "PHL": (7, -8),
           "SGP": (8, -6), "MAC": (-26, 6), "HKG": (-30, -8), "JPN": (7, -10)}
    for cnt, (dx, dy) in lab.items():
        if cnt not in g["CNT"].values:
            continue
        row = g[g["CNT"] == cnt].iloc[0]
        c = RED if cnt == "QCI" else BLUE
        ax.scatter([row["math"]], [row["cmps"]], s=42, c=c, zorder=5,
                   edgecolor="white", linewidth=.8)
        name = "B-S-J-Z, China" if cnt == "QCI" else (
            "Malaysia" if cnt == "MYS" else ("Brunei" if cnt == "BRN" else (
                "Philippines" if cnt == "PHL" else cnt)))
        ax.annotate(name, (row["math"], row["cmps"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, color=c,
                    ha="right" if cnt == "QCI" else "left",
                    weight="bold" if cnt == "QCI" else "normal")
    ax.set_xlabel("Mathematics score")
    ax.set_ylabel("Computational problem solving score")
    ax.set_xlim(g["math"].min() - 20, g["math"].max() + 30)
    fig.savefig(OUT / "fig1_cmps_vs_math.png")
    plt.close(fig)


def fig2(d: pd.DataFrame) -> None:
    """全模型（含位置交互）条件斜率；x 轴为量表原始取值 1–4。"""
    d["x_s"] = d["escs"] * d["z_digital_shortage"]
    d["x_c"] = d["escs"] * d["z_cs_club"]
    d["x_b"] = d["escs"] * d["below_city"]
    need = ["relperf", "escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage",
            "below_city", "x_s", "x_c", "x_b", "CNTSCHID", "w",
            "digital_shortage", "below_city"]
    dd = d.dropna(subset=need).copy()
    m = smf.wls("relperf ~ escs + z_digital_shortage + z_cs_club + z_sw_coverage "
                "+ below_city + x_s + x_c + x_b",
                data=dd, weights=dd["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": dd["CNTSCHID"]})
    V = m.cov_params()
    idx = {k: i for i, k in enumerate(m.params.index)}
    mu = d["digital_shortage"].mean()
    sd = d["digital_shortage"].std()

    def slope(val, below):
        z = (val - mu) / sd
        g = np.zeros(len(m.params))
        g[idx["escs"]] = 1
        g[idx["x_s"]] = z
        if below:
            g[idx["below_city"]] = 1
            g[idx["x_b"]] = z
        b = float(g @ m.params.values)
        se = float(np.sqrt(g @ V.values @ g))
        return b, se

    vals = np.linspace(1, 4, 60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.3),
                                   gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.34})
    for below, c, lab in [(False, BLUE, "City"), (True, RED, "At or below town")]:
        b = np.array([slope(v, below)[0] for v in vals])
        se = np.array([slope(v, below)[1] for v in vals])
        ax1.plot(vals, b, c=c, lw=1.7, label=lab)
        ax1.fill_between(vals, b - 1.96 * se, b + 1.96 * se, color=c, alpha=.14, lw=0)
    ax1.axhline(0, ls="--", c=GREY, lw=.9)
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xlim(1, 4)
    ax1.set_xlabel("School digital-resource shortage (1 = not at all, 4 = a lot)")
    ax1.set_ylabel("ESCS slope on relative overperformance")
    ax1.set_title("(a) Conditional slope", fontsize=9)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")

    x = np.arange(4)
    w = .36
    for k, (below, c, lab) in enumerate([(False, BLUE, "City"),
                                         (True, RED, "At or below town")]):
        b = [slope(v, below)[0] for v in [1, 2, 3, 4]]
        e = [1.96 * slope(v, below)[1] for v in [1, 2, 3, 4]]
        ax2.bar(x + (k - .5) * w, b, width=w, color=c, label=lab, alpha=.9)
        ax2.errorbar(x + (k - .5) * w, b, yerr=e, fmt="none", ecolor="black",
                     capsize=3, lw=.8)
    ax2.axhline(0, ls="--", c=GREY, lw=.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["1", "2", "3", "4"])
    ax2.set_xlabel("Shortage value")
    ax2.set_ylabel("ESCS slope")
    ax2.set_title("(b) At each scale value", fontsize=9)
    fig.savefig(OUT / "fig2_crosslevel.png")
    plt.close(fig)


def fig3(d: pd.DataFrame) -> None:
    dd = d.dropna(subset=["escs", "over_q75"]).copy()
    dd["q"] = dd.groupby("CNT")["escs"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))

    def prof(sub):
        return [sub.loc[sub["q"] == k, "over_q75"].mean() * 100 for k in range(5)]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(range(1, 6), prof(dd), "o-", c=NAVY, lw=1.8, ms=5,
            label=f"All systems (N={len(dd):,})")
    q = dd[dd["CNT"] == "QCI"]
    if len(q):
        ax.plot(range(1, 6), prof(q), "s--", c=RED, lw=1.6, ms=5,
                label=f"B-S-J-Z, China (n={len(q):,})")
    ax.axhline(25, ls=":", c=GREY, lw=.9)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Within-system ESCS quintile (1 = most disadvantaged)")
    ax.set_ylabel("Relative overperformers (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.savefig(OUT / "fig3_escs_quintile.png")
    plt.close(fig)


def fig4() -> None:
    """过程子样本：六项指标的标准化组间差。"""
    d = load()
    p = d[(d["has_ldw"] == 1) & d["relperf"].notna() & d["over_q75"].notna()].copy()
    vars_ = [("time_total_s", "Total time on task"),
             ("actions_mean", "Actions per item"),
             ("ldw_disengaged_ratio", "Disengaged responses"),
             ("first_action_mean_s", "Time to first action"),
             ("transfer_gain", "Transfer gain (Show − Learn)"),
             ("score_per_min", "Score per minute")]
    ov, no = p[p["over_q75"] == 1], p[p["over_q75"] == 0]
    gaps, ses = [], []
    for v, _ in vars_:
        a, b = ov[v].dropna(), no[v].dropna()
        sd = pd.concat([a, b]).std()
        gaps.append((a.mean() - b.mean()) / sd if sd else np.nan)
        ses.append(np.sqrt(a.var() / len(a) + b.var() / len(b)) / sd if sd else np.nan)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = np.arange(len(vars_))[::-1]
    cols = [BLUE if gr > 0 else RED for gr in gaps]
    ax.barh(y, gaps, color=cols, height=.55)
    ax.errorbar(gaps, y, xerr=1.96 * np.array(ses), fmt="none", ecolor="black",
                capsize=3, lw=.9)
    ax.axvline(0, c="black", lw=.9)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in vars_], fontsize=8)
    ax.set_xlabel("Standardised difference (overperformers − others)")
    fig.savefig(OUT / "fig4_process.png")
    plt.close(fig)


def fig5() -> None:
    """体系斜率与 BRR 置信区间。"""
    t = pd.read_csv(ROOT / "output" / "brr_country_slopes.csv").sort_values("b").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    y = np.arange(len(t))
    sig = t["p"] < .05
    lo = t["b"] - 1.96 * t["se_brr"]
    hi = t["b"] + 1.96 * t["se_brr"]
    ax.errorbar(t["b"], y, xerr=[t["b"] - lo, hi - t["b"]], fmt="none",
                ecolor=GREY, lw=.7, alpha=.8)
    ax.scatter(t["b"], y, s=11, c=np.where(sig, NAVY, "white"),
               edgecolor=NAVY, linewidth=.6, zorder=3)
    ax.axvline(0, ls="--", c=RED, lw=.9)
    ax.set_yticks([])
    ax.set_ylabel(f"Systems (n={len(t)}), sorted by slope")
    ax.set_xlabel("ESCS slope on relative overperformance (95% CI, BRR)")
    ax.set_ylim(-2, len(t) + 1)
    ax.text(.98, .04, "filled = interval excludes zero", transform=ax.transAxes,
            ha="right", fontsize=7, color=GREY)
    fig.savefig(OUT / "fig5_country_slopes.png")
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    fig1()
    print("fig1 ok")
    fig2(d)
    print("fig2 ok")
    fig3(d)
    print("fig3 ok")
    fig4()
    print("fig4 ok")
    fig5()
    print("fig5 ok")
