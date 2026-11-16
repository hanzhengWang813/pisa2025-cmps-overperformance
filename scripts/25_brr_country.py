"""
25_brr_country.py — 逐体系 SES 斜率的 BRR 标准误与显著性子类

用途：复核 §5.5 的体系分组（补偿 / 放大 / 不显著）。原分组用的是未接 BRR 的
标准误，边界个案可能在改用复制标准误后改变归类。

做法：对每个体系、每组似真值、每组复制权重，估计 relperf ~ escs 的斜率；
      组内方差 = 1/(80×0.5²)·Σ(θ_r−θ)²，再与 10 组似真值间方差按 Rubin 合并。
      分类阈值：双侧 t（df=9）p<.05。

产出 output/brr_country_slopes.csv、output/brr_country_summary.txt
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
N_PV, NREP, FAY = 10, 80, 0.5
MIN_N = 400


def main() -> None:
    t0 = time.time()
    d = pd.read_csv(ROOT / "data" / "clean" / "analysis.csv.gz",
                    usecols=["CNT", "CNTSTUID", "w", "escs"])
    resid = pd.read_csv(ROOT / "data" / "clean" / "resid.csv.gz",
                        usecols=["CNTSTUID"] + [f"resid{i}" for i in range(1, N_PV + 1)])
    rwt = [f"W_FSTURWT{i}" for i in range(1, NREP + 1)]
    r, _ = pyreadstat.read_sav(str(ROOT / "data/raw/extracted/CY09_MS_STU_PUF.sav"),
                               usecols=["CNTSTUID"] + rwt)
    for x in (d, resid, r):
        x["CNTSTUID"] = x["CNTSTUID"].astype("int64")
    d = d.merge(resid, on="CNTSTUID", how="left").merge(r, on="CNTSTUID", how="left")
    W = np.nan_to_num(np.column_stack([d[c].to_numpy(float) for c in ["w"] + rwt]), nan=0.0)
    print(f"载入 {len(d):,} 行，{time.time()-t0:.0f}s", flush=True)

    keepset = set(pd.read_csv(ROOT / "output" / "tab5_country_slopes.csv")["CNT"])
    print(f"对照：tab5 收录 {len(keepset)} 个体系，仅在其内比较", flush=True)
    cnts = d["CNT"].to_numpy()
    escs = d["escs"].to_numpy(float)
    keep = np.isfinite(escs) & (W[:, 0] > 0)
    out = {}
    for c in pd.unique(cnts):
        m = keep & (cnts == c)
        if c not in keepset or m.sum() < MIN_N:
            continue
        idx = np.where(m)[0]
        y_all = d.iloc[idx][[f"resid{i}" for i in range(1, N_PV + 1)]].to_numpy(float)
        E = escs[idx]
        Wi = W[idx]
        est = np.zeros((N_PV, NREP + 1))
        for p in range(N_PV):
            y = y_all[:, p]
            for rr in range(NREP + 1):
                w = Wi[:, rr]
                ok = np.isfinite(y) & (w > 0)
                if ok.sum() < 50:
                    continue
                x = E[ok]
                X = np.column_stack([np.ones(ok.sum()), x])
                sw = np.sqrt(w[ok])
                b, *_ = np.linalg.lstsq(X * sw[:, None], y[ok] * sw, rcond=None)
                est[p, rr] = b[1]
        full, rep = est[:, 0], est[:, 1:]
        v_brr = float(np.mean(np.sum((rep - full[:, None]) ** 2, axis=1) / (NREP * (1 - FAY) ** 2)))
        qbar = float(full.mean())
        v_pv = (1 + 1 / N_PV) * float(full.var(ddof=1))
        se = float(np.sqrt(v_brr + v_pv))
        t = qbar / se if se > 0 else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t), NREP - 1)) if np.isfinite(t) else np.nan
        out[c] = {"CNT": c, "n": int(m.sum()), "b": qbar, "se": se, "t": t, "p": p,
                  "se_brr": float(np.sqrt(v_brr)), "se_pv": float(np.sqrt(v_pv))}
    print(f"完成 {len(out)} 个体系，{time.time()-t0:.0f}s", flush=True)

    t = pd.DataFrame(out.values()).sort_values("b")
    t["class"] = np.where(t["p"] >= .05, "n.s.",
                          np.where(t["b"] < 0, "compensatory", "amplifying"))
    t.to_csv(ROOT / "output" / "brr_country_slopes.csv", index=False)

    vc = t["class"].value_counts()
    lines = ["=" * 76,
             "逐体系 SES 斜率：BRR 标准误与归类（df = 79，p < .05）",
             "=" * 76,
             f"体系数 {len(t)}；合并斜率（加权平均）{np.average(t['b'], weights=t['n']):+.4f}",
             f"补偿（显著为负）{vc.get('compensatory', 0)}；"
             f"放大（显著为正）{vc.get('amplifying', 0)}；"
             f"不显著 {vc.get('n.s.', 0)}",
             "",
             f"{'CNT':<6}{'n':>7}{'b':>10}{'SE_brr':>9}{'p':>9}   归类"]
    show = pd.concat([t.head(6), t[t.CNT == "QCI"], t.tail(6)])
    for _, r_ in show.iterrows():
        lines.append(f"{r_['CNT']:<6}{int(r_['n']):>7}{r_['b']:>+10.4f}"
                     f"{r_['se_brr']:>9.4f}{r_['p']:>9.4f}   {r_['class']}")
    txt = "\n".join(lines)
    (ROOT / "output" / "brr_country_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
