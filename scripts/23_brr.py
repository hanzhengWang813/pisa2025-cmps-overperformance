"""
23_brr.py — 平衡重复抽样（BRR）标准误（Fay = 0.5，80 组复制权重）

一、为什么此前没有接入
  PISA 2025 学生文件确实带有 80 组复制权重（W_FSTURWT1–80）与 Fay 0.5 的设计。
  但管道分两段：10_build.py 按白名单挑列构建 analysis.csv.gz 时只保留了最终权
  W_FSTUWT（列名 w），复制权重未下行；11_residual.py 又单独产出 resid.csv.gz。
  于是"残差化"和"建模"两段都拿不到 W_FSTURWT，pisa_utils.brr_se 的接口写好却
  没有数据可喂。这不是技术障碍，是推进顺序（先出结果、后补方差）留下的欠账。

二、与论文定义的一致性
  相对表现是一个**定义**而非估计量：国际池化三次多项式回归 + 体系内标准化。
  定义一旦确定，就是数据的一个固定变换。本脚本固定这一步（与 resid.csv.gz 逐位
  一致），只用复制权重重估第二段（分析阶段）的抽样方差，使估计目标与论文同口径。
  若把残差化也逐复制重估，得到的是"加权残差化"这一不同定义下的量，脚本末尾给
  出该口径的对照值。

产出 output/brr_report.txt、output/brr_results.json
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "extracted"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output"

N_PV = 10
NREP = 80
FAY = 0.5
SS = ["escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage", "below_city", "sch_escs"]
NAMES = ["_cons"] + SS + ["escs:z_digital_shortage", "escs:z_cs_club", "escs:below_city"]


def load_data() -> pd.DataFrame:
    usec = set(SS) | {"CNT", "CNTSCHID", "CNTSTUID", "w", "digital_shortage",
                      "cs_club", "sw_coverage"}
    for i in range(1, N_PV + 1):
        for s in ("cmps", "math", "read", "scie"):
            usec.add(f"pv{i}_{s}")
    d = pd.read_csv(CLEAN / "analysis.csv.gz", usecols=lambda c: c in usec)

    rwt = [f"W_FSTURWT{i}" for i in range(1, NREP + 1)]
    t0 = time.time()
    r, _ = pyreadstat.read_sav(str(RAW / "CY09_MS_STU_PUF.sav"), usecols=["CNTSTUID"] + rwt)
    print(f"复制权重 {len(r):,} 行 × {NREP} 组，读入 {time.time()-t0:.0f}s", flush=True)
    r["CNTSTUID"] = r["CNTSTUID"].astype("int64")
    d["CNTSTUID"] = d["CNTSTUID"].astype("int64")
    d = d.merge(r, on="CNTSTUID", how="left")
    print(f"未匹配 {d[rwt[0]].isna().sum():,} 行；总 {len(d):,} 行", flush=True)

    d["sch_escs"] = d.groupby("CNTSCHID")["escs"].transform("mean")
    for v in ["digital_shortage", "cs_club", "sw_coverage"]:
        sd = d[v].std()
        d[f"z_{v}"] = (d[v] - d[v].mean()) / (sd if sd and sd > 0 else 1)
    d["below_city"] = pd.to_numeric(d["below_city"], errors="coerce")
    return d


def design_matrix(sub: pd.DataFrame, ms: list[str]) -> np.ndarray:
    parts = [np.ones((len(sub), 1))]
    for m in ms:
        v = sub[m].to_numpy(dtype=float)
        z = (v - v.mean()) / (v.std() or 1.0)
        parts.append(np.column_stack([z, z ** 2, z ** 3]))
    return np.hstack(parts)


def wls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def combine(mat: np.ndarray) -> dict:
    full, rep = mat[:, 0], mat[:, 1:]
    v_brr = float(np.mean(np.sum((rep - full[:, None]) ** 2, axis=1) / (NREP * (1 - FAY) ** 2)))
    qbar = float(full.mean())
    b = float(full.var(ddof=1))
    v_pv = (1 + 1 / len(full)) * b
    se = float(np.sqrt(v_brr + v_pv))
    df = NREP - 1          # 复制权重决定的参照分布自由度（OECD 2009 手册）
    t = qbar / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df)) if np.isfinite(t) else np.nan
    return {"est": qbar, "se": se, "se_brr": float(np.sqrt(v_brr)),
            "se_pv": float(np.sqrt(v_pv)), "t": float(t), "p": float(p), "df": df}


def main() -> None:
    t_all = time.time()
    d = load_data()
    rwt = [f"W_FSTURWT{i}" for i in range(1, NREP + 1)]
    W = np.nan_to_num(np.column_stack([d[c].to_numpy(dtype=float) for c in ["w"] + rwt]),
                      nan=0.0)
    code, _ = pd.factorize(d["CNT"].to_numpy())
    is_qci = d["CNT"].to_numpy() == "QCI"

    ok_ss = (d[SS].notna().all(axis=1)
             & np.isfinite(d[SS].to_numpy(dtype=float)).all(axis=1)
             & (W[:, 0] > 0)).to_numpy()
    idx_ss = np.where(ok_ss)[0]
    dss = d.iloc[idx_ss]
    X = dss[SS].to_numpy(dtype=float)
    e = X[:, 0]
    X1 = np.column_stack([np.ones(len(X)), e])
    XF = np.column_stack([np.ones(len(X)), X, e * X[:, 1], e * X[:, 2], e * X[:, 4]])
    wss = W[idx_ss]
    qci_ss = dss["CNT"].to_numpy() == "QCI"
    print(f"第二段样本 {len(idx_ss):,} 行 / {dss['CNT'].nunique()} 体系"
          f"（QCI {qci_ss.sum():,}）", flush=True)

    est = {"escs_only": np.zeros((N_PV, NREP + 1)),
           "full": np.zeros((N_PV, NREP + 1, XF.shape[1])),
           "qci_pts": np.zeros((N_PV, NREP + 1)),
           "qci_grad": np.zeros((N_PV, NREP + 1)),
           "all_mean": np.zeros((N_PV, NREP + 1))}

    for m in range(1, N_PV + 1):
        t0 = time.time()
        ycol = f"pv{m}_cmps"
        ms = [f"pv{m}_math", f"pv{m}_read", f"pv{m}_scie"]
        sub = d[[ycol] + ms].dropna()
        row_idx = sub.index.to_numpy()
        Xr = design_matrix(sub, ms)
        yv = sub[ycol].to_numpy(dtype=float)
        # —— 定义固定：未加权三次多项式，与 resid.csv.gz 逐位一致
        beta, *_ = np.linalg.lstsq(Xr, yv, rcond=None)
        res = yv - Xr @ beta
        qci_r = is_qci[row_idx]

        # 体系内未加权 z 标准化（同 11_residual.py）
        g = pd.DataFrame({"r": res, "c": code[row_idx]}).groupby("c", observed=True)["r"]
        z = (res - g.transform("mean").to_numpy()) / g.transform("std").to_numpy()

        inv = np.full(len(d), -1, dtype=np.int64)
        inv[row_idx] = np.arange(len(row_idx))
        src = inv[idx_ss]
        have = src >= 0
        z_ss = np.full(len(idx_ss), np.nan)
        z_ss[have] = z[src[have]]

        for r in range(NREP + 1):
            w_r = wss[:, r]
            ok = np.isfinite(z_ss) & (w_r > 0)
            if ok.sum() < 1000:
                continue
            est["escs_only"][m - 1, r] = wls(z_ss[ok], X1[ok], w_r[ok])[1]
            est["full"][m - 1, r] = wls(z_ss[ok], XF[ok], w_r[ok])
            oq = ok & qci_ss
            if oq.sum() > 100:
                est["qci_grad"][m - 1, r] = wls(z_ss[oq], X1[oq], w_r[oq])[1]
            wr = W[row_idx, r]
            est["qci_pts"][m - 1, r] = np.average(res[qci_r], weights=wr[qci_r])
            est["all_mean"][m - 1, r] = np.average(z, weights=wr)
        print(f"  PV{m} 完成 81 次拟合 {time.time()-t0:.0f}s", flush=True)

    out, R = {}, []
    R.append("=" * 92)
    R.append("BRR 标准误（Fay = 0.5；80 组复制权重；10 组似真值 Rubin 合并）")
    R.append("相对表现的构造固定为论文定义，BRR 只重估分析阶段的抽样方差")
    R.append("=" * 92)

    for key, label, unit in [
            ("qci_pts", "QCI 平均相对表现（CMPS 分值）", "分"),
            ("qci_grad", "QCI 内部 SES 梯度（relperf ~ escs）", "SD/SD"),
            ("all_mean", "全样本平均相对表现（构造上应为 0，校验用）", "z")]:
        c = combine(est[key]); out[key] = c
        R.append(f"\n{label}")
        R.append(f"  est = {c['est']:+.4f} {unit}   SE = {c['se']:.4f} "
                 f"(BRR {c['se_brr']:.4f} + PV {c['se_pv']:.4f})   "
                 f"t = {c['t']:+.2f}   p = {c['p']:.3g}   df = {c['df']:.0f}")

    c1 = combine(est["escs_only"]); out["escs_only"] = c1
    R.append("\nH1：合并样本 SES 梯度（relperf ~ escs，设计基加权）")
    R.append(f"  b = {c1['est']:+.4f}   SE = {c1['se']:.4f} "
             f"(BRR {c1['se_brr']:.4f} + PV {c1['se_pv']:.4f})   "
             f"t = {c1['t']:+.2f}   p = {c1['p']:.3g}")

    R.append("\nM4 全设定（设计基加权，BRR 标准误）")
    R.append(f"  {'term':<26}{'b':>10}{'SE':>9}{'SE_brr':>9}{'t':>8}{'p':>9}")
    out["full"] = {}
    for j, nm in enumerate(NAMES):
        c = combine(est["full"][:, :, j]); out["full"][nm] = c
        star = "***" if c["p"] < .001 else "**" if c["p"] < .01 else "*" if c["p"] < .05 else ""
        R.append(f"  {nm:<26}{c['est']:>+10.4f}{c['se']:>9.4f}{c['se_brr']:>9.4f}"
                 f"{c['t']:>+8.2f}{c['p']:>9.4f} {star}")

    txt = "\n".join(R)
    (OUT / "brr_report.txt").write_text(txt, encoding="utf-8")
    (OUT / "brr_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("\n" + txt)
    print(f"\n总用时 {(time.time()-t_all)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
