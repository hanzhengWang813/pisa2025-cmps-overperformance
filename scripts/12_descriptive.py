"""
12_descriptive.py — 描述性结果（RQ1 与城乡/资源格局）

产出：
  output/tab1_country.csv        体系层汇总
  output/tab2_escs_gradient.csv  ESCS 梯度：CMPS / 学科 / 相对表现 三者对比
  output/tab3_urban.csv          城乡与学校资源格局
  output/descriptive_report.txt  可读报告
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def wmean(x, w):
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.sum() else np.nan


def main() -> None:
    a = pd.read_csv(CLEAN / "analysis.csv.gz")
    r = pd.read_csv(CLEAN / "resid.csv.gz")[["CNTSTUID", "relperf", "over_q75", "over_z1"]]
    d = a.merge(r, on="CNTSTUID", how="left")
    d = d[d["pv1_cmps"].notna() & d["w"].notna() & (d["w"] > 0)].copy()
    print(f"分析样本：{len(d):,} 学生，{d['CNT'].nunique()} 个经济体，"
          f"{d['CNTSCHID'].nunique():,} 所学校")

    # 用第 1 组似真值做描述性（与 OECD 报告惯例一致，正式推断时用 10 组合并）
    d["cmps"] = d["pv1_cmps"]
    d["math"] = d["pv1_math"]
    d["read"] = d["pv1_read"]
    d["scie"] = d["pv1_scie"]

    lines = []
    P = lines.append

    # ================= 表 1：体系层 =================
    rows = []
    for cnt, g in d.groupby("CNT"):
        rows.append({
            "CNT": cnt, "n": len(g),
            "n_sch": g["CNTSCHID"].nunique(),
            "cmps": wmean(g["cmps"], g["w"]),
            "math": wmean(g["math"], g["w"]),
            "read": wmean(g["read"], g["w"]),
            "scie": wmean(g["scie"], g["w"]),
            "escs": wmean(g["escs"], g["w"]),
            "gap_cmps_math": wmean(g["cmps"], g["w"]) - wmean(g["math"], g["w"]),
            "relperf_sd": g["relperf"].std(),
            "over_q75": np.average(g["over_q75"].dropna(),
                                   weights=g.loc[g["over_q75"].notna(), "w"]) if g["over_q75"].notna().any() else np.nan,
            "over_z1": np.average(g["over_z1"].dropna(),
                                  weights=g.loc[g["over_z1"].notna(), "w"]) if g["over_z1"].notna().any() else np.nan,
        })
    t1 = pd.DataFrame(rows).sort_values("cmps", ascending=False)
    t1.to_csv(OUT / "tab1_country.csv", index=False)
    P("=" * 78)
    P("表 1  CMPS 得分前 12 与后 5 的体系（加权，PV1）")
    P("=" * 78)
    P(t1.head(12)[["CNT", "n", "cmps", "math", "gap_cmps_math", "over_q75", "over_z1"]]
      .round(2).to_string(index=False))
    P("")
    P(t1.tail(5)[["CNT", "n", "cmps", "math", "gap_cmps_math", "over_q75", "over_z1"]]
      .round(2).to_string(index=False))
    qci = t1[t1["CNT"] == "QCI"].iloc[0]
    P(f"\n中国四省市（QCI）：CMPS {qci['cmps']:.1f}，数学 {qci['math']:.1f}，"
      f"CMPS−数学 {qci['gap_cmps_math']:+.1f}；"
      f"超额者比例 q75 {qci['over_q75']*100:.1f}%，z≥1 {qci['over_z1']*100:.1f}%")
    P(f"QCI 的 CMPS 排名：{list(t1['CNT']).index('QCI')+1} / {len(t1)}")

    # ================= 表 2：ESCS 梯度 =================
    P("\n" + "=" * 78)
    P("表 2  ESCS 的标准化梯度：CMPS 与学科、相对表现的对比（H1 核心证据）")
    P("=" * 78)
    grad = []
    for cnt, g in d.groupby("CNT"):
        sub = g.dropna(subset=["escs", "cmps", "math", "relperf"])
        if len(sub) < 400 or sub["escs"].std() < 1e-6:
            continue
        rec = {"CNT": cnt, "n": len(sub)}
        for v in ["cmps", "math", "read", "scie", "relperf"]:
            x = sub["escs"].to_numpy(float)
            y = sub[v].to_numpy(float)
            w = sub["w"].to_numpy(float)
            # 加权标准化斜率
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
            sy = np.sqrt(np.average((y - my) ** 2, weights=w))
            cov = np.average((x - mx) * (y - my), weights=w)
            rec[v] = cov / (sx * sy) * (sy / sy) if sx > 0 else np.nan  # 标准化 beta
            rec[v] = cov / (sx ** 2) / sy if sx > 0 else np.nan  # β 折算为 y 的 SD 单位
        grad.append(rec)
    t2 = pd.DataFrame(grad)
    t2.to_csv(OUT / "tab2_escs_gradient.csv", index=False)
    P(f"覆盖 {len(t2)} 个经济体（每体系 n≥400 且 ESCS 有变异）")
    P("\n各梯度均值（ESCS 每上升 1 个标准差，因变量变动的标准差单位）：")
    P(t2[["cmps", "math", "read", "scie", "relperf"]].describe().loc[["mean", "std", "min", "max"]]
      .round(3).to_string())
    P(f"\nCMPS 梯度均值 {t2['cmps'].mean():.3f}，数学 {t2['math'].mean():.3f}，"
      f"相对表现 {t2['relperf'].mean():.3f}")
    P(f"→ 相对表现的 SES 梯度仅为 CMPS 原始梯度的 "
      f"{t2['relperf'].mean()/t2['cmps'].mean()*100:.1f}%，"
      f"为数学梯度的 {t2['relperf'].mean()/t2['math'].mean()*100:.1f}%")
    n_neg = (t2["relperf"] < 0).sum()
    P(f"→ {n_neg}/{len(t2)} 个体系的相对表现梯度为负或不显著异于零")
    if "QCI" in set(t2["CNT"]):
        qq = t2[t2["CNT"] == "QCI"].iloc[0]
        P(f"\nQCI：CMPS {qq['cmps']:.3f} / 数学 {qq['math']:.3f} / "
          f"相对表现 {qq['relperf']:.3f}")

    # ================= 表 3：城乡与学校资源 =================
    P("\n" + "=" * 78)
    P("表 3  城乡位置、学校数字资源与相对表现")
    P("=" * 78)
    rows = []
    for lab, g in d.groupby("urban3"):
        ok = g.dropna(subset=["relperf"])
        rows.append({
            "城乡": lab,
            "n": len(ok),
            "超额率q75": ok["over_q75"].mean(),
            "relperf均值": ok["relperf"].mean(),
            "数字资源短缺": g["digital_shortage"].mean(),
            "计算机俱乐部": g["cs_club"].mean(),
            "软件覆盖": g["sw_coverage"].mean(),
            "ESCS均值": g["escs"].mean(),
        })
    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "tab3_urban.csv", index=False)
    P(t3.round(3).to_string(index=False))
    P("\n注：urban3 中 rural=村庄/小城镇，town=城镇，city=城市及以上")

    # QCI 内部
    q = d[d["CNT"] == "QCI"]
    P(f"\n--- 中国四省市内部（N={len(q):,}）---")
    for lab in ["rural", "town", "city"]:
        g = q[q["urban3"] == lab]
        ok = g.dropna(subset=["relperf"])
        if len(ok) == 0:
            continue
        P(f"  {lab:<6} n={len(ok):>5}  超额率 {ok['over_q75'].mean()*100:5.1f}%  "
          f"relperf {ok['relperf'].mean():+.3f}  ESCS {g['escs'].mean():+.2f}  "
          f"数字短缺 {g['digital_shortage'].mean():.2f}  "
          f"俱乐部 {g['cs_club'].mean():.2f}")

    # ================= ESCS 五分位上的超额率 =================
    P("\n" + "=" * 78)
    P("表 4  ESCS 五分位上的超额者比例（全样本，加权）")
    P("=" * 78)
    dd = d.dropna(subset=["escs", "over_q75"]).copy()
    dd["q"] = dd.groupby("CNT")["escs"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    for lab, g in dd.groupby("q"):
        P(f"  第 {int(lab)+1} 五分位  n={len(g):>7,}  超额率 {g['over_q75'].mean()*100:5.1f}%  "
          f"relperf {g['relperf'].mean():+.3f}")

    txt = "\n".join(lines)
    (OUT / "descriptive_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
