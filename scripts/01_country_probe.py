"""
01_country_probe.py — 国家覆盖探针

目的：直接从已下载完整的学校文件中读出 CNT / CNTRYID 的全部取值映射，
确认中国（B-S-J-Z）在 PISA 2025 数据中的实际编码，并统计其学校数。
这是方案第七节"待核验项 1/2"的前置步骤。
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pyreadstat

BASE = pathlib.Path(__file__).resolve().parent.parent
EXT = BASE / "data" / "raw" / "extracted"
OUT = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)

sav = EXT / "CY09_MS_SCH_PUF.sav"
if not sav.exists():
    raise SystemExit(f"未找到 {sav}，请先运行 00_inspect.py")

df, meta = pyreadstat.read_sav(
    str(sav),
    usecols=["CNT", "CNTRYID", "SC001Q01TA", "STRATUM", "OECD"],
)

print(f"学校记录数: {len(df):,}")
print(f"CNT 唯一值: {df['CNT'].nunique()}   CNTRYID 唯一值: {df['CNTRYID'].nunique()}\n")

# 值标签映射：把数字代码翻成国名
lab_cnt = meta.variable_value_labels.get("CNT", {})
lab_id = meta.variable_value_labels.get("CNTRYID", {})

rows = []
for cnt, n in df["CNT"].value_counts().sort_index().items():
    name = lab_cnt.get(cnt, "")
    cid = ""
    sub = df.loc[df["CNT"] == cnt, "CNTRYID"]
    if len(sub):
        cid = sub.iloc[0]
    name_id = lab_id.get(cid, "")
    rows.append({"CNT": cnt, "CNTRYID": cid, "国名_CNT标签": name, "国名_ID标签": name_id, "学校数": n})

tbl = pd.DataFrame(rows)
tbl.to_csv(OUT / "country_coverage_school.csv", index=False, encoding="utf-8-sig")

print("=== 全部国家/经济体（按 CNT 排序）===")
with pd.option_context("display.max_rows", 200, "display.width", 150):
    print(tbl.to_string(index=False))

# 定向搜索中国
print("\n=== 中国相关条目 ===")
mask = (
    tbl["国名_CNT标签"].astype(str).str.contains("china|chinese", case=False, na=False)
    | tbl["国名_ID标签"].astype(str).str.contains("china|chinese", case=False, na=False)
    | tbl["CNT"].astype(str).str.upper().isin(["CHN", "CHNBSJZ"])
)
hit = tbl[mask]
print(hit.to_string(index=False) if len(hit) else "未匹配到 China 字面，请人工核对上表")

# 若定位到中国，看其学校社区类型分布
if len(hit):
    for cnt in hit["CNT"].unique():
        sub = df[df["CNT"] == cnt]
        print(f"\n=== CNT={cnt} 的学校社区类型分布（SC001Q01TA）===")
        lab_sc = meta.variable_value_labels.get("SC001Q01TA", {})
        vc = sub["SC001Q01TA"].value_counts().sort_index()
        for k, v in vc.items():
            print(f"  {k}  {lab_sc.get(k, ''):<28} {v:>5} 所")
