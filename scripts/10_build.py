"""
10_build.py — 合并学生、学校、过程数据，生成分析用主数据集

数据源：
  CY09_MS_STU_PUF.sav        学生问卷（CMPS 似真值、学科似真值、ESCS、权重）
  CY09_MS_SCH_PUF.sav        学校问卷（数字资源、城乡）
  CY09_MS_LDW_*.sav          LDW 题目级过程数据（已由 06 汇总为 ldw_process.csv.gz）

产出 data/clean/analysis.csv.gz

关键编码处理：
  - PISA 量表题 1..k 有效，>=95 为 valid skip / system missing / not administered /
    invalid / no response，一律置缺失
  - ESCS 等 WLE 指数按原值使用
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pyreadstat

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "extracted"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

STU = RAW / "CY09_MS_STU_PUF.sav"


def clean_likert(s: pd.Series, k: int) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    v[(v < 1) | (v > k)] = np.nan
    return v


def main() -> None:
    pv_c = [f"PV{i}CMPS" for i in range(1, 11)]
    pv_m = [f"PV{i}MATH" for i in range(1, 11)]
    pv_r = [f"PV{i}READ" for i in range(1, 11)]
    pv_s = [f"PV{i}SCIE" for i in range(1, 11)]

    use = (["CNT", "CNTSCHID", "CNTSTUID", "W_FSTUWT", "ESCS", "HISEI",
            "PAREDINT", "HOMEPOS", "ST004D01T", "ST001D01T", "Option_LDW"]
           + pv_c + pv_m + pv_r + pv_s
           + ["ST250Q02JA", "ST250Q05JA", "ST254Q02JA", "ST254Q03JA",
              "ST438Q01DA", "ST438Q02DA", "ST438Q03DA", "ST438Q04DA",
              "AIUSESCH", "PQDIGEFF"])

    print("读取学生文件…", flush=True)
    df, meta = pyreadstat.read_sav(str(STU), usecols=use)
    print(f"  {len(df):,} 行 × {df.shape[1]} 列")

    out = pd.DataFrame({
        "CNT": df["CNT"].astype(str),
        "CNTSCHID": pd.to_numeric(df["CNTSCHID"], errors="coerce").astype("Int64"),
        "CNTSTUID": pd.to_numeric(df["CNTSTUID"], errors="coerce").astype("Int64"),
        "w": pd.to_numeric(df["W_FSTUWT"], errors="coerce"),
    })

    for pref, lst in [("cmps", pv_c), ("math", pv_m), ("read", pv_r), ("scie", pv_s)]:
        arr = df[lst].apply(pd.to_numeric, errors="coerce")
        for i, c in enumerate(lst, start=1):
            out[f"pv{i}_{pref}"] = arr[c].to_numpy()

    out["escs"] = pd.to_numeric(df["ESCS"], errors="coerce")
    out["hisei"] = pd.to_numeric(df["HISEI"], errors="coerce")
    out["paredint"] = pd.to_numeric(df["PAREDINT"], errors="coerce")
    out["homepos"] = pd.to_numeric(df["HOMEPOS"], errors="coerce")
    out["female"] = clean_likert(df["ST004D01T"], 2).map({1.0: 1.0, 2.0: 0.0})
    out["grade"] = pd.to_numeric(df["ST001D01T"], errors="coerce")
    out["grade"] = out["grade"].where(out["grade"].between(1, 13))

    # 家庭数字资源：1=有, 2=无
    out["home_computer"] = clean_likert(df["ST250Q02JA"], 2).map({1.0: 1.0, 2.0: 0.0})
    out["home_internet"] = clean_likert(df["ST250Q05JA"], 2).map({1.0: 1.0, 2.0: 0.0})
    # 家庭设备数量（1=无, 2=1台, 3=2台, 4=3台及以上）
    out["n_desktop"] = clean_likert(df["ST254Q02JA"], 4)
    out["n_laptop"] = clean_likert(df["ST254Q03JA"], 4)

    # AI 聊天机器人使用频次（4 种用途）
    ai = pd.concat([clean_likert(df[c], 5) for c in
                    ["ST438Q01DA", "ST438Q02DA", "ST438Q03DA", "ST438Q04DA"]], axis=1)
    out["ai_use_school"] = ai.iloc[:, 0]
    out["ai_use_mean"] = ai.mean(axis=1, skipna=True)
    out["aiusesch"] = pd.to_numeric(df["AIUSESCH"], errors="coerce")
    out["pqdigeff"] = pd.to_numeric(df["PQDIGEFF"], errors="coerce")

    # ---------------- 学校层 ----------------
    sch = pd.read_csv(CLEAN / "school_index.csv.gz")
    sch = sch[["CNT", "CNTSCHID", "digital_shortage", "digital_quality",
               "cs_club", "sw_coverage", "urban3", "urban6", "below_city"]]
    out = out.merge(sch, on=["CNT", "CNTSCHID"], how="left")

    # ---------------- 过程数据 ----------------
    proc = pd.read_csv(CLEAN / "ldw_process.csv.gz")
    keep = ["CNTSTUID", "has_ldw", "ldw_items_total", "ldw_score",
            "ldw_disengaged_ratio", "ldw_notreached_ratio",
            "score_learn", "score_apply", "score_show", "transfer_gain",
            "time_total_s", "time_mean_s", "rapid_ratio", "actions_mean",
            "first_action_mean_s", "score_per_min"]
    out = out.merge(proc[keep], on="CNTSTUID", how="left")

    print(f"\n合并后：{len(out):,} 行 × {out.shape[1]} 列")
    print("CMPS 有效：", int(out["pv1_cmps"].notna().sum()))
    print("过程子样本：", int((out["has_ldw"] == 1).sum()))
    print("QCI：", int((out["CNT"] == "QCI").sum()),
          " 其中过程子样本：", int(((out["CNT"] == "QCI") & (out["has_ldw"] == 1)).sum()))

    print("\n关键变量缺失率（%）")
    chk = ["pv1_cmps", "pv1_math", "escs", "w", "female", "grade", "home_computer",
           "ai_use_school", "digital_shortage", "cs_club", "urban3"]
    for c in chk:
        print(f"  {c:<18} {out[c].isna().mean()*100:5.1f}%")

    # 保留有 CMPS 且有权重者进入分析
    ok = out["pv1_cmps"].notna() & out["w"].notna() & (out["w"] > 0)
    print(f"\n可通过筛选的样本：{int(ok.sum()):,}")
    print("未通过者经济体分布（前5）：")
    print(out.loc[~ok, "CNT"].value_counts().head().to_string())

    p = CLEAN / "analysis.csv.gz"
    out.to_csv(p, index=False, compression="gzip")
    print(f"\n已写出 {p}")


if __name__ == "__main__":
    main()
