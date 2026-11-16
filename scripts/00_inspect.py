"""
00_inspect.py — PISA 2025 数据结构勘察（不加载全量数据）

目的：在读取任何数据行之前，只解析 SPSS 元数据，核验方案中的三个未决项：
  1. 中国（CNT=CHN）是否出现在 学生 / LDW / 教师 / 过程 四类文件中
  2. LDW 似真值（plausible values）的实际变量命名
  3. 认知过程数据的字段粒度：题目级汇总，还是事件级（点击/按键序列）

方法：pyreadstat 以 metadataonly=True 读取，只取列名与标签，内存占用极低。

用法：
    cd <repository root>
    python3 scripts/00_inspect.py

输出：
    output/inspect_report.txt     人类可读报告
    output/columns_<file>.csv     每个文件的完整变量清单（名 + 标签）
"""

from __future__ import annotations

import pathlib
import re
import sys
import traceback
import zipfile

import pandas as pd
import pyreadstat

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
EXT = RAW / "extracted"
OUT = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)

FILES = {
    "STU": "CY09_MS_STU_PUF.zip",
    "SCH": "CY09_MS_SCH_PUF.zip",
    "LDW": "CY09_MS_LDW_PUF.zip",
    "TCH": "CY09_MS_TCH_PUF.zip",
    "PROC": "CY09_MS_COG_PROCESS_PUF.zip",
}

# 关键词：用来从上千个变量里捞出我们关心的那些
PATTERNS = {
    "国家标识": r"^(CNT|CNTRYID|COUNTRY)$",
    "学生/学校ID": r"(CNTSTUID|CNTSCHID|CNTTCHID|SUBJECT)",
    "学生权重": r"^W_FSTUWT$",
    "复制权重(BRR)": r"^W_FSTURWT\d+$|^W_FSTR\d+$",
    "分层/初级抽样": r"^(STRATUM|VAR_UNIT|OECD|NC)$",
    "社会经济地位": r"^ESCS$|^SCSCES$|ESCS",
    "学校社区类型": r"^SC001",
    "似真值": r"^PV\d*_?(LDW|CPR|MATH|READ|SCIE)|^(LDW|CPR|MATH|READ|SCIE)PV\d+|^PV\d+[A-Z]{2,}",
    "LDW相关": r"LDW",
    "过程数据线索": r"(TIME|DURATION|VISIT|RETRY|HELP|ACTION|EVENT|SEQ|CLICK|KEYSTROKE|TIMESTAMP|ATTEMPT|HINT)",
    "教师问卷": r"^(T\d|TC|TT|TCH)",
}


def extract(zip_path: pathlib.Path) -> list[str]:
    """解压并返回其中的 sav 文件名清单。已解压则跳过。"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".sav")]
        for n in names:
            target = EXT / pathlib.Path(n).name
            if not target.exists():
                z.extract(n, EXT)
                # 若 zip 内含子目录，挪到扁平位置
                extracted = EXT / n
                if extracted != target and extracted.exists():
                    extracted.rename(target)
        return [pathlib.Path(n).name for n in names]


def inspect_sav(sav: pathlib.Path, tag: str, report: list[str]) -> None:
    df_meta, meta = pyreadstat.read_sav(str(sav), metadataonly=True)

    cols = list(meta.column_names)
    labels = dict(zip(meta.column_names, meta.column_labels))

    report.append("=" * 78)
    report.append(f"[{tag}] {sav.name}")
    report.append(f"  变量总数: {len(cols):,}   行数(元数据): {meta.number_rows:,}")
    report.append("=" * 78)

    # 变量清单落盘
    pd.DataFrame({"variable": cols, "label": [labels.get(c, "") for c in cols]}).to_csv(
        OUT / f"columns_{tag}.csv", index=False, encoding="utf-8-sig"
    )

    for name, pat in PATTERNS.items():
        hits = [c for c in cols if re.search(pat, c, flags=re.IGNORECASE)]
        if hits:
            shown = hits[:25]
            more = f"  ...(共 {len(hits)} 个)" if len(hits) > 25 else ""
            report.append(f"\n  ▸ {name}  ({len(hits)})")
            for c in shown:
                lab = labels.get(c, "") or ""
                lab = lab[:60]
                report.append(f"      {c:<18} {lab}")
            if more:
                report.append(f"      {more}")

    # 值标签里找国家代码（若存在 CNT 变量）
    for cand in ("CNT", "CNTRYID", "COUNTRY"):
        if cand in meta.variable_value_labels:
            vv = meta.variable_value_labels[cand]
            hits = {k: v for k, v in vv.items() if str(v).upper().startswith(("CHN", "CHINA"))}
            if hits:
                report.append(f"\n  ▸ 中国在国家变量 {cand} 中的取值: {hits}")
            else:
                report.append(f"\n  ▸ 国家变量 {cand} 共 {len(vv)} 个取值，未匹配到 CHN/China 字面")
    report.append("")


def main() -> int:
    report: list[str] = [
        "PISA 2025 数据结构勘察报告",
        f"数据目录: {RAW}",
        "",
    ]

    found_any = False
    for tag, fname in FILES.items():
        zp = RAW / fname
        if not zp.exists() or zp.stat().st_size < 1000:
            report.append(f"[{tag}] 缺失或未下载完整: {fname}\n")
            continue
        try:
            savs = extract(zp)
            report.append(f"[{tag}] 解压得到: {savs}")
            for s in savs:
                p = EXT / s
                if p.exists():
                    inspect_sav(p, tag, report)
                    found_any = True
        except Exception:
            report.append(f"[{tag}] 处理失败:\n{traceback.format_exc()}\n")

    text = "\n".join(report)
    (OUT / "inspect_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if found_any else 1


if __name__ == "__main__":
    sys.exit(main())
