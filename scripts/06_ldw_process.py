"""
06_ldw_process.py — 从 LDW 题目级记录构建过程指标（修正版）

编码事实（来自变量取值标签，已实测确认）：
  S  计分：0=无分 1-4=部分分 5=满分；7=Not Applicable；9=Disengaged（放弃）；r=Not reached
  TT 总用时：毫秒（中位数约 14 万 ms ≈ 140 秒/题）
  A  操作次数：整数；99999997/8/9 为 Not Applicable / Invalid / No Response
  F  首次操作前停留：毫秒；同上缺失码

设计事实：LDW 为轮转设计，每名学生只作答 6 个单元中的 2 个（约 1/3 学生被抽到 LDW）。

产出 data/clean/ldw_process.csv.gz
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
import pyreadstat

ROOT = pathlib.Path(__file__).resolve().parent.parent
# OECD 的 LDW 文件名带发布批次日期（如 CY09_MS_LDW_20260806.sav），故按前缀匹配
_cands = sorted((ROOT / "data" / "raw" / "extracted").glob("CY09_MS_LDW*.sav"))
LDW = _cands[0] if _cands else ROOT / "data" / "raw" / "extracted" / "CY09_MS_LDW_20260806.sav"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

UNIT_NAME = {1: "Recycling Claw", 2: "Fitness App", 3: "Nature Pod",
             5: "Safe Landing", 6: "Art Exhibition", 7: "Digital Art"}
MISSING_CUT = 9999997  # 99999997/8/9 等缺失码的下界
PAT = re.compile(r"^P1M(\d)(\d\d)(S|TT|A|F|R)$")


def item_meta(cols, labels) -> pd.DataFrame:
    recs = []
    for c in cols:
        m = PAT.match(str(c))
        if not m:
            continue
        lab = str(labels.get(c, ""))
        st = re.search(r"(Learn|Apply|Show)\s*(\d+)", lab)
        recs.append({"name": c, "unit": int(m.group(1)), "kind": m.group(3),
                     "stage": st.group(1) if st else None})
    return pd.DataFrame(recs)


def main() -> None:
    print("读取 LDW 文件…", flush=True)
    df, meta = pyreadstat.read_sav(str(LDW))
    labels = dict(zip(meta.column_names, meta.column_labels))
    IM = item_meta(df.columns, labels)
    print(f"学生 {len(df):,}；题目级变量 {len(IM)}", flush=True)

    out = pd.DataFrame({
        "CNT": df["CNT"].astype(str),
        "CNTSCHID": pd.to_numeric(df["CNTSCHID"], errors="coerce").astype("Int64"),
        "CNTSTUID": pd.to_numeric(df["CNTSTUID"], errors="coerce").astype("Int64"),
    })

    def names(kind, stage=None):
        d = IM[(IM["kind"] == kind) & IM["stage"].notna()]
        if stage:
            d = d[d["stage"] == stage]
        return d["name"].tolist()

    # ---------------- 计分 ----------------
    S_cols = names("S")
    Sraw = df[S_cols]
    # 统一转字符后判定：0-5 有效分；9 放弃；r 未到达；7 不适用
    Sstr = Sraw.astype("string")
    Snum = Sstr.apply(pd.to_numeric, errors="coerce")
    valid = Snum.where((Snum >= 0) & (Snum <= 5))

    # 按题目自身满分归一：Show 阶段题目为二分（满分 1），Learn/Apply 为多级（满分 2-5）。
    # 若统一除以 5，阶段之间与题目之间都不可比，故逐题除以该题实测最高分。
    item_max = valid.max(axis=0).replace(0, np.nan)
    norm = valid.div(item_max, axis=1).clip(0, 1)
    print("题目满分分布：")
    print(item_max.value_counts().sort_index().to_string())

    out["ldw_items_total"] = norm.notna().sum(axis=1)
    out["ldw_score"] = norm.mean(axis=1, skipna=True)
    out["ldw_disengaged_n"] = (Snum == 9).sum(axis=1)
    out["ldw_notreached_n"] = (Sstr == "r").sum(axis=1)
    denom = out["ldw_items_total"] + out["ldw_disengaged_n"] + out["ldw_notreached_n"]
    out["ldw_disengaged_ratio"] = out["ldw_disengaged_n"] / denom.replace(0, np.nan)
    out["ldw_notreached_ratio"] = out["ldw_notreached_n"] / denom.replace(0, np.nan)

    # 单元层
    unit_of = dict(zip(IM["name"], IM["unit"]))
    for u, uname in UNIT_NAME.items():
        cs = [c for c in S_cols if unit_of.get(c) == u]
        sub = norm[cs]
        out[f"u{u}_n"] = sub.notna().sum(axis=1)
        out[f"u{u}_score"] = sub.mean(axis=1, skipna=True)
    out["ldw_units_done"] = sum(
        (out[f"u{u}_n"] >= 1).astype(int) for u in UNIT_NAME
    )

    # 阶段层
    for st in ["Learn", "Apply", "Show"]:
        cs = names("S", st)
        sub = norm[cs]
        out[f"score_{st.lower()}"] = sub.mean(axis=1, skipna=True)
        out[f"n_{st.lower()}"] = sub.notna().sum(axis=1)
    out["transfer_gain"] = out["score_show"] - out["score_learn"]

    # ---------------- 时间（毫秒 → 秒） ----------------
    TT_cols = names("TT")
    TT = df[TT_cols].apply(pd.to_numeric, errors="coerce")
    TT = TT.where((TT >= 0) & (TT < MISSING_CUT)) / 1000.0
    out["time_total_s"] = TT.sum(axis=1, skipna=True)
    out["time_mean_s"] = TT.mean(axis=1, skipna=True)
    out["n_timed"] = TT.notna().sum(axis=1)
    out["rapid_ratio"] = (TT < 5).sum(axis=1).div(TT.notna().sum(axis=1).replace(0, np.nan))

    # ---------------- 操作次数 ----------------
    A = df[names("A")].apply(pd.to_numeric, errors="coerce")
    A = A.where((A >= 0) & (A < MISSING_CUT))
    out["actions_mean"] = A.mean(axis=1, skipna=True)
    out["actions_total"] = A.sum(axis=1, skipna=True)
    out["zero_action_ratio"] = (A == 0).sum(axis=1).div(A.notna().sum(axis=1).replace(0, np.nan))

    # ---------------- 首操作潜伏期（毫秒 → 秒） ----------------
    F = df[names("F")].apply(pd.to_numeric, errors="coerce")
    F = F.where((F >= 0) & (F < MISSING_CUT)) / 1000.0
    out["first_action_mean_s"] = F.mean(axis=1, skipna=True)

    # ---------------- 效率：每题每秒得分 ----------------
    out["score_per_min"] = np.where(
        out["time_total_s"] > 30,
        out["ldw_score"] * out["ldw_items_total"] / (out["time_total_s"] / 60.0),
        np.nan,
    )

    # 只保留真正参与了 LDW 的学生（至少有 1 道有效计分题或 1 道放弃标记）
    has_ldw = (out["ldw_items_total"] >= 1) | (out["ldw_disengaged_n"] >= 1)
    out["has_ldw"] = has_ldw.astype(int)
    print(f"参与 LDW 的学生：{int(has_ldw.sum()):,} / {len(out):,} "
          f"({has_ldw.mean()*100:.1f}%)")

    core = ["ldw_items_total", "ldw_score", "ldw_disengaged_ratio", "ldw_notreached_ratio",
            "score_learn", "score_apply", "score_show", "transfer_gain",
            "time_total_s", "time_mean_s", "rapid_ratio", "actions_mean",
            "zero_action_ratio", "first_action_mean_s", "score_per_min"]
    part = out[has_ldw]
    print("\n=== 全样本（参与者）描述 ===")
    print(part[core].describe().T.round(3).to_string())

    q = part[part["CNT"] == "QCI"]
    print(f"\n=== 中国四省市 QCI（N={len(q):,}）===")
    print(q[core].describe().T.round(3).to_string())
    print("\n单元数分布：", q["ldw_units_done"].value_counts().sort_index().to_dict())

    p = CLEAN / "ldw_process.csv.gz"
    out.to_csv(p, index=False, compression="gzip")
    print(f"\n已写出 {p}（{len(out):,} 行 × {out.shape[1]} 列）")


if __name__ == "__main__":
    main()
