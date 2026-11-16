"""
03_find_vars.py — 变量自动发现器

不预设 PISA 的变量命名（似真值、权重、ID 的命名历年并不统一），
而是用正则从元数据中实测定位，输出可信的变量清单。

定位目标：
  1. 抽样设计：CNT / CNTSCHID / CNTSTUID / STRATUM / OECD
  2. 权重：学生最终权重、80 组 BRR 复制权重
  3. 似真值：LDW（因变量）、数学、阅读、科学
  4. 背景：ESCS、学校社区类型
  5. 过程数据线索：题目级 vs 事件级
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd
import pyreadstat

BASE = pathlib.Path(__file__).resolve().parent.parent
EXT = BASE / "data" / "raw" / "extracted"
OUT = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "ID_与抽样": [r"^CNT$", r"^CNTRYID$", r"^CNTSCHID$", r"^CNTSTUID$", r"^STRATUM$", r"^OECD$"],
    "学生权重": [r"^W_FSTUWT$"],
    "BRR复制权重": [r"^W_FSTURWT\d+$"],
    "似真值_LDW": [r"^PV\d+_?LDW", r"^LDW_?PV\d+", r"^PV\d+LDW"],
    "似真值_数学": [r"^PV\d+_?MATH", r"^MATH_?PV\d+", r"^PV\d+MATH"],
    "似真值_阅读": [r"^PV\d+_?READ", r"^READ_?PV\d+", r"^PV\d+READ"],
    "似真值_科学": [r"^PV\d+_?SCIE", r"^SCIE_?PV\d+", r"^PV\d+SCIE"],
    "ESCS": [r"^ESCS$"],
    "学校社区类型": [r"^SC001Q01TA$"],
    "过程数据线索": [r"(DURATION|TIME_ON|VISIT|ATTEMPT|RETRY|HINT|HELP|ACTION|EVENT|SEQUENCE|CLICK|TIMESTAMP)"],
}

summary = []

for sav in sorted(EXT.glob("*.sav")):
    _, meta = pyreadstat.read_sav(str(sav), metadataonly=True)
    cols = list(meta.column_names)
    labels = dict(zip(cols, meta.column_labels))

    print("=" * 88)
    print(f"{sav.name}   变量 {len(cols):,}   行 {meta.number_rows:,}")
    print("=" * 88)

    for name, pats in TARGETS.items():
        hits = []
        for pat in pats:
            hits += [c for c in cols if re.search(pat, c, flags=re.IGNORECASE)]
        hits = sorted(set(hits))
        if hits:
            preview = hits[:12]
            tail = f"  ...(共 {len(hits)})" if len(hits) > 12 else ""
            print(f"  ▸ {name:<12} {len(hits):>3} 个: {', '.join(preview)}{tail}")
            summary.append(
                {"file": sav.name, "target": name, "n": len(hits), "vars": ",".join(hits)}
            )
    print()

if summary:
    pd.DataFrame(summary).to_csv(OUT / "var_discovery.csv", index=False, encoding="utf-8-sig")
    print(f"已写出 {OUT / 'var_discovery.csv'}")
