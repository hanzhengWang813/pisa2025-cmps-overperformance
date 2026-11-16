"""
05_stu_probe.py — 学生问卷变量探测（仅读元数据，快）

目的：不靠猜测命名，用正则从真实变量标签里定位分析所需变量：
  - 抽样设计与权重（W_FSTUWT / W_FSTURWT*）
  - 家庭社会经济文化地位（ESCS 及其三个成分）
  - 学科似真值（数学 / 阅读 / 科学）
  - 性别、年级、移民背景
  - 数字设备、数字活动、AI 使用、数字自我效能

输出：output/stu_probe.txt
"""

from __future__ import annotations

import pathlib
import re

import pyreadstat

ROOT = pathlib.Path(__file__).resolve().parent.parent
STU = ROOT / "data" / "raw" / "extracted" / "CY09_MS_STU_PUF.sav"
OUT = ROOT / "output" / "stu_probe.txt"

PATTERNS = {
    "权重/设计": r"^(W_|VSRW|STRRAT|VAR_UNIT|OECD|CNT$|CNTSCHID|CNTSTUID)",
    "ESCS/家庭背景": r"(ESCS|HISEI|PARED|HOMEPOS|CULTPOS|WEALTH|IMMIG|LANGN|FAMSUP)",
    "学科似真值": r"(PV\d+(MATH|READ|SCIE)|^PV\d+(MATH|READ|SCIE))",
    "性别年级": r"^(ST00[1-4]|ST004|GRADE|ST001|ST002|ST003)",
    "数字设备/资源": r"(ICTRES|ICTHOME|ICTSCH|COMP|TABLET|LAPTOP|SMARTPHONE|INTERNET|DEVICE|DIGITAL)",
    "数字活动/AI": r"(AI|ARTIFICIAL|CHATBOT|PROGRAM|CODING|COMPUT|ROBOT|SIMUL|DATA)",
    "自我效能/态度": r"(EFFIC|SELF|CONFID|MOTIV|ANXI|CURIOS|PERSIST|GROWTH|ATTIT)",
    "学习时间/策略": r"(STUDY|HOMEWORK|TIME|STRATEG|METACOG|PLAN|MONITOR)",
}


def main() -> None:
    if not STU.exists():
        raise SystemExit(f"未找到 {STU}")
    _, meta = pyreadstat.read_sav(str(STU), metadataonly=True)
    cols = meta.column_names
    labels = dict(zip(cols, meta.column_labels))

    lines = [f"学生文件变量总数：{len(cols)}", ""]
    seen: set[str] = set()

    for theme, pat in PATTERNS.items():
        hits = [c for c in cols if re.search(pat, c + " " + str(labels.get(c, "")), re.I)]
        # 权重类变量太多，只列前若干个代表
        if theme == "权重/设计":
            hits = hits[:12] + [h for h in hits if h.startswith("W_FSTURWT")][:3]
        lines.append(f"--- {theme}：{len(hits)} 个 ---")
        for c in hits:
            if c in seen:
                continue
            seen.add(c)
            lines.append(f"  {c:<18} {str(labels.get(c, ''))[:88]}")
        lines.append("")

    # 每个变量的取值标签（用于确认李克特方向）
    lines.append("=== 关键变量的取值标签 ===")
    for c in cols:
        if c in ("ST004D01T", "ST001D01T", "ESCS"):
            lines.append(f"{c}: {meta.variable_value_labels.get(c, {})}")

    txt = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    print(txt[:6000])
    print(f"\n完整清单已写出 {OUT}")


if __name__ == "__main__":
    main()
