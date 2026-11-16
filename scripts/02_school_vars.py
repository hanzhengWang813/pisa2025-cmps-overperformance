"""
02_school_vars.py — 从学校问卷中定位学校层分析变量

不重读数据，直接消费 00_inspect.py 产出的变量清单（output/columns_SCH.csv），
按研究主题归类，为方案第六节的学校层指标提供实证依据。

主题：
  A. 数字基础设施与 ICT 资源      -> 学校层核心自变量
  B. 数字/信息科技课程与活动供给    -> 机会结构
  C. 教师与资源短缺                -> 补偿能力边界
  D. 学校人口学与组织特征           -> 控制变量
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "output"

cols = pd.read_csv(OUT / "columns_SCH.csv")
cols["label"] = cols["label"].fillna("")

THEMES = {
    "A_数字基础设施与ICT资源": r"(computer|ict|digital|internet|device|tablet|laptop|software|wifi|broadband|printer|projector|smartboard)",
    "B_数字课程与活动供给": r"(curricul|course|programm|program|activity|club|extracurric|computational|coding|informatic)",
    "C_教师与资源短缺": r"(shortage|short|lack|inadequat|scarce|staff|teacher)",
    "D_学校特征与控制": r"(school type|ownership|funding|source|size|enrol|class size|grade|urban|rural|community|location|principal|management|accountab|assess)",
}

report = ["学校问卷变量主题勘察（共 %d 个变量）\n" % len(cols)]

for theme, pat in THEMES.items():
    mask = cols["variable"].str.contains(pat, case=False, na=False) | cols["label"].str.contains(
        pat, case=False, na=False
    )
    sub = cols[mask].copy()
    report.append("=" * 86)
    report.append(f"{theme}   命中 {len(sub)} 个变量")
    report.append("=" * 86)
    if len(sub):
        for _, r in sub.iterrows():
            lab = str(r["label"])[:78]
            report.append(f"  {r['variable']:<16} {lab}")
    report.append("")

text = "\n".join(report)
(OUT / "school_vars_thematic.txt").write_text(text, encoding="utf-8")
print(text)

# 同时落一份机器可读的分类表
frames = []
for theme, pat in THEMES.items():
    mask = cols["variable"].str.contains(pat, case=False, na=False) | cols["label"].str.contains(
        pat, case=False, na=False
    )
    sub = cols[mask].copy()
    sub["theme"] = theme
    frames.append(sub)
pd.concat(frames).to_csv(OUT / "school_vars_thematic.csv", index=False, encoding="utf-8-sig")
