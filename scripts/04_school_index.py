"""
04_school_index.py — 构建学校层指标（基于 PISA 2025 学校问卷）

产出 data/clean/school_index.csv.gz（仅含 CNT / CNTSCHID 与学校层派生变量）

指标设计：
  digital_shortage   数字资源短缺（SC017Q09JA，越大越短缺）
  digital_quality    数字资源质量差（SC017Q10JA，越大越差）
  cs_club            是否提供计算机/编程俱乐部（SC053Q08TA，1=是）
  soft_index         软机会指数 = cs_club 等数字活动供给的标准化合成
  urban6 / urban3    社区类型（6 类原始 / 3 类合并）
  sch_escs           学校平均 ESCS（由学生文件回填，本脚本仅留占位）

PISA 缺失编码约定：1..k 为有效作答，>=5（或 >=k+1）为 valid skip / not administered /
no response / invalid 等无效值，必须置为缺失，否则会被误当最差等级。
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

SCH = RAW / "CY09_MS_SCH_PUF.sav"


def clean_likert(s: pd.Series, k: int) -> pd.Series:
    """把 1..k 视为有效，其余（含 5..9 编码）置缺失。"""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s[(s < 1) | (s > k)] = np.nan
    return s


def build() -> pd.DataFrame:
    cols = [
        "CNT", "CNTSCHID", "SC001Q01TA",
        "SC017Q09JA", "SC017Q10JA",
        "SC053Q08TA",
        "SC261Q01DA", "SC261Q02DA", "SC261Q03DA", "SC261Q04DA",
        "SC261Q05DA", "SC261Q06DA", "SC261Q07DA", "SC261Q08DA",
        "SC261Q09DA", "SC261Q10DA",
    ]
    df, meta = pyreadstat.read_sav(str(SCH), usecols=cols)
    print(f"学校文件：{len(df):,} 行，{df['CNT'].nunique()} 个经济体")

    out = pd.DataFrame({
        "CNT": df["CNT"].astype(str),
        "CNTSCHID": df["CNTSCHID"].astype("int64"),
    })

    # --- 数字资源短缺 / 质量（4 级李克特）---
    out["digital_shortage"] = clean_likert(df["SC017Q09JA"], 4)
    out["digital_quality"] = clean_likert(df["SC017Q10JA"], 4)

    # --- 计算机/编程俱乐部（1=有, 2=无）---
    club = pd.to_numeric(df["SC053Q08TA"], errors="coerce").astype(float)
    club[~club.isin([1.0, 2.0])] = np.nan
    out["cs_club"] = club.map({1.0: 1.0, 2.0: 0.0})

    # --- 设备软件覆盖（0-100 百分比）---
    sw = df[[f"SC261Q{i:02d}DA" for i in range(1, 11)]].apply(
        pd.to_numeric, errors="coerce"
    )
    sw = sw.where((sw >= 0) & (sw <= 100))
    out["sw_coverage"] = sw.mean(axis=1, skipna=True)
    out["sw_n_available"] = sw.notna().sum(axis=1)

    # --- 城乡 ---
    loc = pd.to_numeric(df["SC001Q01TA"], errors="coerce").astype(float)
    loc[~loc.isin([1, 2, 3, 4, 5, 6])] = np.nan
    out["urban6"] = loc
    out["urban3"] = loc.map({1: "rural", 2: "rural", 3: "town",
                             4: "city", 5: "city", 6: "city"})
    out["rural"] = out["urban3"].map({"rural": 1.0, "town": np.nan, "city": 0.0})
    # 县镇及以下（rural + town）
    out["below_city"] = loc.map({1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0})

    q = out[out["CNT"] == "QCI"]
    print(f"\nQCI 学校 {len(q)} 所")
    print("城乡分布：")
    print(q["urban3"].value_counts(dropna=False).to_string())
    print("\n指标缺失率（QCI）")
    for c in ["digital_shortage", "digital_quality", "cs_club", "sw_coverage"]:
        print(f"  {c:<18} {q[c].isna().mean()*100:5.1f}%")
    return out


if __name__ == "__main__":
    d = build()
    p = CLEAN / "school_index.csv.gz"
    d.to_csv(p, index=False, compression="gzip")
    print(f"\n已写出 {p}  ({len(d):,} 行 × {d.shape[1]} 列)")
