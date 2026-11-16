"""
24_brr_china.py — 中国子样本（QCI）两张关键结果的 BRR 标准误

对应论文附录 B 与 5.5 节：
  1. QCI 内部 SES 梯度（含性别协变量）
  2. QCI 性别差距（女 − 男）
  3. QCI 平均相对表现（CMPS 分值）

方法同 23_brr.py：相对表现的构造固定为论文定义，BRR 只重估分析阶段方差。
产出 output/brr_china.txt
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("brr", ROOT / "scripts" / "23_brr.py")
brr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brr)

N_PV, NREP, FAY = 10, 80, brr.FAY


def main() -> None:
    d = pd.read_csv(ROOT / "data" / "clean" / "analysis.csv.gz",
                    usecols=["CNT", "CNTSTUID", "w", "escs", "female"])
    rwt = [f"W_FSTURWT{i}" for i in range(1, NREP + 1)]
    import pyreadstat
    r, _ = pyreadstat.read_sav(str(ROOT / "data/raw/extracted/CY09_MS_STU_PUF.sav"),
                               usecols=["CNTSTUID"] + rwt)
    r["CNTSTUID"] = r["CNTSTUID"].astype("int64")
    d["CNTSTUID"] = d["CNTSTUID"].astype("int64")
    d = d.merge(r, on="CNTSTUID", how="left")
    W = np.nan_to_num(np.column_stack([d[c].to_numpy(float) for c in ["w"] + rwt]), nan=0.0)

    resid = pd.read_csv(ROOT / "data" / "clean" / "resid.csv.gz",
                        usecols=["CNTSTUID"] + [f"resid{i}" for i in range(1, N_PV + 1)])
    resid["CNTSTUID"] = resid["CNTSTUID"].astype("int64")
    d = d.merge(resid, on="CNTSTUID", how="left")

    q = d["CNT"].to_numpy() == "QCI"
    idx = np.where(q)[0]
    dq = d.iloc[idx]
    X = np.column_stack([np.ones(len(dq)), dq["escs"].to_numpy(float),
                         dq["female"].to_numpy(float)])
    ok0 = np.isfinite(X).all(axis=1) & (W[idx, 0] > 0)
    print(f"QCI 样本 {len(idx):,}，模型可用 {ok0.sum():,}", flush=True)

    est = {k: np.zeros((N_PV, NREP + 1)) for k in ["b_escs", "b_female"]}
    for m in range(1, N_PV + 1):
        y = dq[f"resid{m}"].to_numpy(float)
        for r_ in range(NREP + 1):
            w = W[idx, r_]
            ok = ok0 & np.isfinite(y) & (w > 0)
            if ok.sum() < 100:
                continue
            b = brr.wls(y[ok], X[ok], w[ok])
            est["b_escs"][m - 1, r_] = b[1]
            est["b_female"][m - 1, r_] = b[2]

    lines = ["=" * 88, "中国（B-S-J-Z）子样本：BRR 标准误（Fay = 0.5，80 组复制权重）", "=" * 88]
    for k, lab in [("b_escs", "ESCS 梯度（控制性别）"), ("b_female", "女性 vs 男性（控制 ESCS）")]:
        c = brr.combine(est[k])
        star = "***" if c["p"] < .001 else "**" if c["p"] < .01 else "*" if c["p"] < .05 else ""
        lines.append(f"\n{lab}")
        lines.append(f"  b = {c['est']:+.4f}   SE = {c['se']:.4f} "
                     f"(BRR {c['se_brr']:.4f} + PV {c['se_pv']:.4f})   "
                     f"t = {c['t']:+.2f}   p = {c['p']:.4f} {star}   df = {c['df']:.0f}")
    txt = "\n".join(lines)
    (ROOT / "output" / "brr_china.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
