"""
13_models.py — 多层模型与跨层交互（H1–H4 的正式检验）

模型设定（因变量为体系内标准化的相对表现 relperf，即"学业基线之外"的部分）：

  M1  null            relperf ~ 1                       （估计 ICC）
  M2  baseline        relperf ~ escs                    （H1：SES 梯度是否延伸到超额）
  M3  school          + 学校层数字资源与城乡            （RQ3 主效应）
  M4  crosslevel      + escs × 学校层变量               （H3/H4：补偿还是放大）
  M5  process         relperf ~ 过程指标（过程子样本）  （RQ2：策略还是噪声）
  M6  lpm             over_q75 ~ M4 设定                （比例解读，便于政策表述）

估计策略：
  - 10 组似真值分别拟合多层模型（学校随机截距），系数按 Rubin 规则合并
  - 另以加权最小二乘 + 学校聚类稳健标准误作设计基（design-based）对照，
    因 statsmodels 的 MixedLM 不支持抽样权重
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

N_PV = 10


def load() -> pd.DataFrame:
    a = pd.read_csv(CLEAN / "analysis.csv.gz")
    r = pd.read_csv(CLEAN / "resid.csv.gz")
    r = r[["CNTSTUID", "relperf"] + [f"resid{i}" for i in range(1, N_PV + 1)] +
          ["over_q75", "over_z1"]]
    d = a.merge(r, on="CNTSTUID", how="inner")
    # 学校层均值 ESCS（作为学校社会经济构成的控制）
    d["sch_escs"] = d.groupby("CNTSCHID")["escs"].transform("mean")
    # 学校层数字资源标准化，便于解释交互
    for v in ["digital_shortage", "digital_quality", "cs_club", "sw_coverage"]:
        sd = d[v].std()
        d[f"z_{v}"] = (d[v] - d[v].mean()) / (sd if sd and sd > 0 else 1)
    d["below_city"] = pd.to_numeric(d["below_city"], errors="coerce")
    return d


def prep(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = d.dropna(subset=cols).copy()
    x = x[np.isfinite(x[cols]).all(axis=1)]
    return x


def fit_mixed(d: pd.DataFrame, y: str, rhs: str, extra_drop: list[str]):
    cols = [y, "CNTSCHID"] + extra_drop
    x = prep(d, cols)
    m = smf.mixedlm(f"{y} ~ {rhs}", data=x, groups=x["CNTSCHID"]).fit(reml=True)
    return m, len(x)


def pool_over_pv(d: pd.DataFrame, rhs: str, extra: list[str], label: str,
                 var_of_interest: list[str]) -> dict:
    """对 10 组残差分别拟合，Rubin 合并指定系数。"""
    res = {v: {"est": [], "var": []} for v in var_of_interest}
    ns, vc, rs = [], [], []
    for i in range(1, N_PV + 1):
        y = f"resid{i}"
        m, n = fit_mixed(d, y, rhs, [y] + extra)
        ns.append(n)
        vc.append(float(m.cov_re.iloc[0, 0]))
        rs.append(float(m.scale))
        for v in var_of_interest:
            if v in m.params.index:
                res[v]["est"].append(float(m.params[v]))
                res[v]["var"].append(float(m.bse[v]) ** 2)
    out = {"label": label, "n": int(np.mean(ns)),
           "icc": float(np.mean(vc) / (np.mean(vc) + np.mean(rs))),
           "coef": {}}
    for v in var_of_interest:
        if not res[v]["est"]:
            continue
        e, va = np.array(res[v]["est"]), np.array(res[v]["var"])
        qbar = e.mean()
        ubar = va.mean()
        b = e.var(ddof=1)
        tvar = ubar + (1 + 1 / len(e)) * b
        se = np.sqrt(tvar)
        from scipy import stats
        t = qbar / se if se > 0 else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t), len(e) - 1)) if np.isfinite(t) else np.nan
        out["coef"][v] = {"b": float(qbar), "se": float(se), "t": float(t), "p": float(p)}
    return out


def ols_check(d: pd.DataFrame, y: str, rhs: str, var: str) -> dict:
    """加权 OLS + 学校聚类稳健标准误（设计基对照）。"""
    terms = [t.strip() for t in rhs.replace(":", "+").replace("*", "+").split("+")]
    x = prep(d, [y, "CNTSCHID", "w"] + terms)
    m = smf.wls(f"{y} ~ {rhs}", data=x, weights=x["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": x["CNTSCHID"]})
    return {"n": int(m.nobs), "b": float(m.params.get(var, np.nan)),
            "se": float(m.bse.get(var, np.nan)), "p": float(m.pvalues.get(var, np.nan))}


def show(r: dict) -> str:
    s = f"\n--- {r['label']}  (N≈{r['n']:,}, ICC={r['icc']:.4f}) ---"
    for v, c in r["coef"].items():
        star = "***" if c["p"] < .001 else "**" if c["p"] < .01 else "*" if c["p"] < .05 else ""
        s += f"\n  {v:<34} b={c['b']:+.4f}  SE={c['se']:.4f}  p={c['p']:.4f} {star}"
    return s


def main() -> None:
    d = load()
    print(f"载入 {len(d):,} 行；relperf 有效 {d['relperf'].notna().sum():,}；"
          f"学校 {d['CNTSCHID'].nunique():,}")

    # 分析样本：有相对表现、ESCS、学校层变量
    base = prep(d, ["relperf", "escs", "CNTSCHID"])
    print(f"基础分析样本 {len(base):,}（{base['CNT'].nunique()} 个经济体）", flush=True)

    report = []
    R = report.append
    R("=" * 84)
    R("多层模型结果（因变量：体系内标准化相对表现 relperf；10 组似真值 Rubin 合并）")
    R("=" * 84)

    out_json = {}

    # ---------------- M1 零模型 ----------------
    m0, n0 = fit_mixed(base, "relperf", "1", ["relperf"])
    vc = float(m0.cov_re.iloc[0, 0]); rs = float(m0.scale)
    R(f"\n[M1 零模型] N={n0:,}  学校层方差={vc:.4f}  学生层方差={rs:.4f}  "
      f"ICC={vc/(vc+rs):.4f}")
    R(f"→ 相对表现的组内相关系数为 {vc/(vc+rs)*100:.1f}%，"
      f"即约 {vc/(vc+rs)*100:.0f}% 的变异位于学校之间")
    out_json["M1"] = {"n": n0, "icc": vc / (vc + rs), "var_school": vc, "var_student": rs}

    # ---------------- M2 ESCS 基线 ----------------
    print("running r2...", flush=True)
    r2 = pool_over_pv(base, "escs", ["relperf", "escs"], "M2 基线：ESCS → 相对表现", ["escs"])
    R(show(r2))
    out_json["M2"] = r2

    # ---------------- M3 学校层 ----------------
    school_terms = "z_digital_shortage + z_cs_club + z_sw_coverage + below_city + sch_escs"
    drop3 = ["relperf", "escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage",
             "below_city", "sch_escs"]
    print("running r3...", flush=True)
    r3 = pool_over_pv(base, "escs + " + school_terms, drop3,
                      "M3 学校层主效应",
                      ["escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage",
                       "below_city", "sch_escs"])
    R(show(r3))
    out_json["M3"] = r3

    # ---------------- M4 跨层交互 ----------------
    inter = ("escs:z_digital_shortage + escs:z_cs_club + escs:below_city")
    print("running r4...", flush=True)
    r4 = pool_over_pv(base, "escs + " + school_terms + " + " + inter, drop3,
                      "M4 跨层交互（补偿 vs 放大）",
                      ["escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage",
                       "below_city", "sch_escs",
                       "escs:z_digital_shortage", "escs:z_cs_club", "escs:below_city"])
    R(show(r4))
    out_json["M4"] = r4

    # ---------------- M6 线性概率模型：是否超额 ----------------
    d_lpm = d.dropna(subset=["over_q75", "escs", "z_digital_shortage", "z_cs_club",
                             "z_sw_coverage", "below_city", "sch_escs"]).copy()
    lpm_rhs = "escs + " + school_terms + " + " + inter
    vars_lpm = ["escs", "z_digital_shortage", "z_cs_club", "z_sw_coverage", "below_city",
                "sch_escs", "escs:z_digital_shortage", "escs:z_cs_club", "escs:below_city"]
    m = smf.wls(f"over_q75 ~ {lpm_rhs}", data=d_lpm, weights=d_lpm["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": d_lpm["CNTSCHID"]})
    R(f"\n--- M6 线性概率模型（因变量：是否处于体系内相对表现前 25%）N={int(m.nobs):,} ---")
    out_json["M6"] = {}
    for v in vars_lpm:
        b, se, p = float(m.params[v]), float(m.bse[v]), float(m.pvalues[v])
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        R(f"  {v:<34} b={b*100:+.2f}pp  SE={se*100:.2f}  p={p:.4f} {star}")
        out_json["M6"][v] = {"b": b, "se": se, "p": p}

    # ---------------- M5 过程机制 ----------------
    R("\n" + "=" * 84)
    R("M5 过程机制（过程子样本：有 LDW 题目级记录者）")
    R("=" * 84)
    proc = d[d["has_ldw"] == 1].copy()
    proc["z_time"] = (proc["time_total_s"] - proc["time_total_s"].mean()) / proc["time_total_s"].std()
    proc["z_actions"] = (proc["actions_mean"] - proc["actions_mean"].mean()) / proc["actions_mean"].std()
    R(f"过程子样本 N={len(proc):,}，其中 relperf 有效 {proc['relperf'].notna().sum():,}")
    proc_vars = ["z_time", "z_actions", "ldw_disengaged_ratio", "first_action_mean_s"]

    # 注意：score_learn / score_show 是 LDW 原始表现的组成部分，与因变量同源，
    # 不得作为预测变量进入模型，否则构成同义反复。此处只用行为过程指标。
    # 机制模型改用加权最小二乘 + 学校聚类稳健标准误（MixedLM 在过程子样本上
    # 因学校间方差较大而不收敛，且此处核心量是固定效应，OLS 估计一致）。
    rhs_p = "escs + " + " + ".join(proc_vars)
    step_cols = ["relperf", "escs", "CNTSCHID", "w"] + proc_vars
    pv_all = proc.dropna(subset=step_cols).copy()
    R(f"可用 N={len(pv_all):,}")

    m5 = smf.wls(f"relperf ~ {rhs_p}", data=pv_all, weights=pv_all["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": pv_all["CNTSCHID"]})
    R(f"\n--- M5 过程指标 → 相对表现  (N={int(m5.nobs):,}, R²={m5.rsquared:.4f}) ---")
    out_json["M5"] = {"n": int(m5.nobs), "r2": float(m5.rsquared), "coef": {}}
    for v in ["escs"] + proc_vars:
        b, se, p = float(m5.params[v]), float(m5.bse[v]), float(m5.pvalues[v])
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        R(f"  {v:<34} b={b:+.4f}  SE={se:.4f}  p={p:.4f} {star}")
        out_json["M5"]["coef"][v] = {"b": b, "se": se, "p": p}

    # 逐步加入，观察解释力变化
    R("\n  逐步回归（单层 WLS，仅作解释力参考）：")
    step_specs = [
        ("仅 ESCS", "escs"),
        ("+ 用时", "escs + z_time"),
        ("+ 操作次数", "escs + z_time + z_actions"),
        ("+ 放弃率", "escs + z_time + z_actions + ldw_disengaged_ratio"),
        ("+ 首操作潜伏期", "escs + z_time + z_actions + ldw_disengaged_ratio + first_action_mean_s"),
        ("+ 迁移增益", "escs + z_time + z_actions + ldw_disengaged_ratio + first_action_mean_s + transfer_gain"),
    ]
    out_json["M5_step"] = {}
    for lab, rhs in step_specs:
        mm = smf.wls(f"relperf ~ {rhs}", data=pv_all, weights=pv_all["w"]).fit()
        R(f"    {lab:<16} R²={mm.rsquared:.4f}  adj.R²={mm.rsquared_adj:.4f}")
        out_json["M5_step"][lab] = float(mm.rsquared_adj)

    # ---------------- 设计基对照（加权 + 学校聚类） ----------------
    R("\n" + "=" * 84)
    R("设计基对照：加权最小二乘 + 学校聚类稳健标准误")
    R("=" * 84)
    out_json["design_based"] = {}
    for lab, rhs, var in [
        ("ESCS → 相对表现", "escs", "escs"),
        ("ESCS → 相对表现（含学校层与交互）", lpm_rhs, "escs"),
        ("ESCS×数字资源短缺", lpm_rhs, "escs:z_digital_shortage"),
        ("ESCS×计算机俱乐部", lpm_rhs, "escs:z_cs_club"),
        ("ESCS×县镇及以下", lpm_rhs, "escs:below_city"),
    ]:
        o = ols_check(d, "relperf", rhs, var)
        star = "***" if o["p"] < .001 else "**" if o["p"] < .01 else "*" if o["p"] < .05 else ""
        R(f"  {lab:<32} b={o['b']:+.4f} SE={o['se']:.4f} p={o['p']:.4f} {star}  (N={o['n']:,})")
        out_json["design_based"][lab] = o

    txt = "\n".join(report)
    (OUT / "models_report.txt").write_text(txt, encoding="utf-8")
    (OUT / "models_results.json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
