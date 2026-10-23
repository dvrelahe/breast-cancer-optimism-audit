"""Additional analyses for case study 1 and Fig. 1.

Run in the same folder after both notebooks: CaseStudy1_MetastasisLadder.ipynb writes geo/compendium_expr.npz,
geo/compendium_clinical.csv and casestudy1_results.json, and PECT_DigitalTwin_Audit_v3.ipynb writes twin_v3_results.json.
This script produces:
  1) 95% bootstrap CIs for the external calibration slopes of penalised logistic regression under strict L1
     (Table IV; 2,000 resamples of patients within each held-out cohort). The point estimates are checked against
     casestudy1_results.json and the intervals are saved in calibration_ci.json;
  2) the comparison of included and excluded patients (censored before 60 months) on every recorded clinical variable;
  3) Fig. 1 (600 dpi) from casestudy1_results.json and twin_v3_results.json.
Fig. 2 is drawn by the last cell of PECT_DigitalTwin_Audit_v3.ipynb.
"""
import gzip, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, rankdata
from sklearn.linear_model import LogisticRegression

G = "geo"; SEED = 20260806; B = 2000

# ---------------------------------------------------------------- shared helpers (identical to the notebook)
def qn_reference(X): return np.sort(X, axis=1).mean(axis=0)
def qn_apply(X, ref): return ref[np.argsort(np.argsort(X, axis=1), axis=1)]
def combat_fit(X, batch):
    b_ids = np.unique(batch); grand = X.mean(0); var_pooled = ((X - grand) ** 2).mean(0) + 1e-8
    S = (X - grand) / np.sqrt(var_pooled); params = {}
    gh = np.array([S[batch == b].mean(0) for b in b_ids]); dh = np.array([S[batch == b].var(0, ddof=1) + 1e-8 for b in b_ids])
    gbar = gh.mean(1); t2 = gh.var(1, ddof=1)
    a_prior = (2 * dh.var(1, ddof=1) + dh.mean(1) ** 2) / dh.var(1, ddof=1); b_prior = (dh.mean(1) * dh.var(1, ddof=1) + dh.mean(1) ** 3) / dh.var(1, ddof=1)
    for k, b in enumerate(b_ids):
        Sb = S[batch == b]; nb = Sb.shape[0]; g_old, d_old = gh[k].copy(), dh[k].copy()
        for _ in range(50):
            g_new = (t2[k] * nb * gh[k] + d_old * gbar[k]) / (t2[k] * nb + d_old)
            d_new = (0.5 * ((Sb - g_new) ** 2).sum(0) + b_prior[k]) / (nb / 2 + a_prior[k] - 1)
            if np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)) < 1e-4 and np.max(np.abs(d_new - d_old) / d_old) < 1e-4:
                g_old, d_old = g_new, d_new; break
            g_old, d_old = g_new, d_new
        params[b] = (g_old, d_old)
    return dict(grand=grand, var=var_pooled, params=params)
def combat_apply(X, batch, fit):
    S = (X - fit["grand"]) / np.sqrt(fit["var"]); out = S.copy()
    for b in np.unique(batch):
        if b in fit["params"]:
            g, d = fit["params"][b]; out[batch == b] = (S[batch == b] - g) / np.sqrt(d)
    return out * np.sqrt(fit["var"]) + fit["grand"]
class Prep:
    def fit(self, X, batch):
        self.ref = qn_reference(X); Xq = qn_apply(X, self.ref)
        self.cb = combat_fit(Xq, batch) if len(np.unique(batch)) > 1 else None
        Xc = combat_apply(Xq, batch, self.cb) if self.cb else Xq
        self.mu, self.sd = Xc.mean(0), Xc.std(0) + 1e-6; return self
    def transform(self, X, batch):
        Xq = qn_apply(X, self.ref); Xc = combat_apply(Xq, batch, self.cb) if self.cb else Xq
        return (Xc - self.mu) / self.sd
def select_genes(X, y, k=50):
    R = np.apply_along_axis(rankdata, 0, X); n1 = y.sum(); n0 = len(y) - n1
    return np.argsort(-np.abs((R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5))[:k]
def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]

# ---------------------------------------------------------------- 1) calibration slopes with bootstrap CIs
def calibration_cis():
    Z = np.load(f"{G}/compendium_expr.npz", allow_pickle=True); XALL = Z["X"].astype(np.float64)
    CLIN = pd.read_csv(f"{G}/compendium_clinical.csv", index_col=0).loc[Z["samples"]]
    stored = json.load(open("casestudy1_results.json")); out = {}
    for tname, col in [("DM5", "dm5"), ("BONE", "bone")]:
        m = CLIN[col].notna().values; X, y, coh = XALL[m], CLIN[col].values[m].astype(int), CLIN.cohort.values[m]
        preds = np.zeros(len(y))
        for c in dict.fromkeys(coh):
            te = coh == c; tr = ~te; pp = Prep().fit(X[tr], coh[tr])
            Xtr, Xte = pp.transform(X[tr], coh[tr]), pp.transform(X[te], coh[te]); g = select_genes(Xtr, y[tr])
            preds[te] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(Xtr[:, g], y[tr]).predict_proba(Xte[:, g])[:, 1]
        rng = np.random.default_rng(SEED); out[tname] = {}
        for c in dict.fromkeys(coh):
            yc, pc = y[coh == c], preds[coh == c]; s = calib(yc, pc); boots = []
            for _ in range(B):
                i = rng.integers(0, len(yc), len(yc))
                if 0 < yc[i].sum() < len(i): boots.append(calib(yc[i], pc[i]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            assert abs(s - stored[tname]["calibration"]["Penalised LR"][c]["slope"]) < 1e-6   # same estimate as the notebook
            out[tname][c] = dict(n=int(len(yc)), events=int(yc.sum()), slope=float(s), ci=[float(lo), float(hi)])
            print(f"{tname} {c}: slope {s:.2f} (95% CI {lo:.2f} to {hi:.2f}); n = {len(yc)}, events = {yc.sum()}")
    json.dump(out, open("calibration_ci.json", "w"), indent=1)

# ---------------------------------------------------------------- 2) included vs excluded patients
def read_matrix(acc):
    meta = {}
    with gzip.open(f"{G}/{acc}_series_matrix.txt.gz", "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"): break
            if line.startswith("!Sample_"):
                meta.setdefault(line.split("\t")[0], []).append([v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]])
    char = {}
    for vals in meta.get("!Sample_characteristics_ch1", []):
        for j, v in enumerate(vals):
            if ":" in v: k, x = v.split(":", 1); char.setdefault(k.strip(), [None] * len(vals))[j] = x.strip()
    clin = pd.DataFrame(char); clin["source"] = meta["!Sample_source_name_ch1"][0]; return clin
num = lambda s: pd.to_numeric(s.replace({"--": np.nan, "?": np.nan, "NA": np.nan}), errors="coerce")
def dm5(t, e):
    y = pd.Series(np.nan, index=t.index); y[t >= 60] = 0; y[(e == 1) & (t <= 60)] = 1; return y
def compare(name, v, inc, kind):
    a, b = v[inc].dropna(), v[~inc].dropna()
    if kind == "cont":
        p = mannwhitneyu(a, b).pvalue; print(f"  {name}: included median {a.median():.1f} (n={len(a)}) vs excluded {b.median():.1f} (n={len(b)}), Mann-Whitney p = {p:.3f}")
    else:
        lv = sorted(set(a) | set(b)); tab = np.array([[(a == l).sum() for l in lv], [(b == l).sum() for l in lv]])
        p = fisher_exact(tab).pvalue if tab.shape == (2, 2) else chi2_contingency(tab)[1]
        print(f"  {name}: included {dict(zip(lv, tab[0].tolist()))} vs excluded {dict(zip(lv, tab[1].tolist()))}, p = {p:.3f}")
def excluded_vs_included():
    cl = read_matrix("GSE11121"); inc = dm5(num(cl["t.dmfs"]), num(cl["e.dmfs"])).notna()
    print(f"GSE11121: {inc.sum()} included, {(~inc).sum()} excluded")
    compare("tumour size (cm)", num(cl["size_in_cm"]), inc, "cont"); compare("grade", num(cl["grade"]), inc, "cat")
    cl = read_matrix("GSE7390"); inc = dm5(num(cl["t.dmfs"]) / 30.4375, num(cl["e.dmfs"])).notna()
    print(f"GSE7390: {inc.sum()} included, {(~inc).sum()} excluded")
    for v, k in [("age", "cont"), ("size", "cont"), ("grade", "cat"), ("er", "cat")]: compare(v, num(cl[v]), inc, k)
    cl = read_matrix("GSE2603")
    cl = cl[cl.source.str.contains("tumor|tumour|primary", case=False) & ~cl.source.str.contains("cell line|MDA|MCF", case=False)]
    inc = num(cl["binary 5y met event"]).notna(); print(f"GSE2603: {inc.sum()} included, {(~inc).sum()} excluded")
    for v in ["age at dx", "tumor size (cm)", "pos lymph nodes"]: compare(v, num(cl[v]), inc, "cont")
    er = cl["path er status"].where(cl["path er status"].isin(["N", "P"])); compare("ER status", er, inc, "cat")

# ---------------------------------------------------------------- 3) Fig. 1
FS = 8; W, H = 3.45, 3.45 * 819 / 1380
COLS = ["#3b7d4f", "#3b6ea5", "#e0a030", "#cc5a5a", "#8c2d2d"]
def _style(): plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans", "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})
def fig1(out="Fig1_600dpi.png"):
    _style(); cs = json.load(open("casestudy1_results.json")); tw = json.load(open("twin_v3_results.json"))
    light = lambda c, a=0.5: tuple(np.array(to_rgb(c)) + (1 - np.array(to_rgb(c))) * a)
    def ladder(t):   # bars: mean AUC; error bars: L1 = random-effects CI averaged over families, L2-L5 = released ladder_range
        d = cs[t]; L = d["ladder"]; R = d["ladder_range"]; m = list(d["L1"]["strict"])
        v = [L["L1_strict"], L["L2"], L["L3"], L["L4"], L["L5"]]
        lo = [np.mean([d["L1"]["strict"][k]["meta"]["lo"] for k in m])] + [R[k][0] for k in ["L2", "L3", "L4", "L5"]]
        hi = [np.mean([d["L1"]["strict"][k]["meta"]["hi"] for k in m])] + [R[k][1] for k in ["L2", "L3", "L4", "L5"]]
        return np.array(v), np.array(lo), np.array(hi)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw=dict(width_ratios=[1.45, 1])); x = np.arange(5); w = 0.38
    for k, t in enumerate(["DM5", "BONE"]):
        v, lo, hi = ladder(t); xx = x - w / 2 + k * w
        a.bar(xx, v, width=w, color=[c if k == 0 else light(c) for c in COLS], edgecolor="white", lw=0.4)
        a.errorbar(xx, v, yerr=[v - lo, hi - v], fmt="none", ecolor="black", lw=0.7, capsize=1.8, capthick=0.7)
    a.axhline(0.5, ls=":", color="gray", lw=0.8); a.set_ylim(0.45, 1.12); a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_xticks(x); a.set_xticklabels(["L1", "L2", "L3", "L4", "L5"]); a.set_ylabel("AUC (higher = better)"); a.set_title("(a) Metastasis classifiers", fontsize=FS)
    a.text(0.03, 0.98, "solid: 5-yr distant metastasis\nlight: bone relapse", transform=a.transAxes, va="top", fontsize=FS - 1.5)
    lad = tw["ladder"]; md = np.array([r["MdAPE"] for r in lad]); clo = np.array([r["CI_lo"] for r in lad]); chi = np.array([r["CI_hi"] for r in lad])
    b.bar(x, md, color=COLS, width=0.65); b.errorbar(x, md, yerr=[md - clo, chi - md], fmt="none", ecolor="black", lw=0.7, capsize=1.8, capthick=0.7)
    st = [r for r in tw["table"] if r["protocol"] == "L1" and r["key"] == "static"][0]["MdAPE"]
    b.axhline(st, ls="--", color="gray", lw=0.8); b.text(4.6, st + 1.2, "lesion-only\nrate", ha="right", va="bottom", color="gray", fontsize=FS - 1.5, linespacing=0.95)
    b.set_ylim(0, 55); b.set_xticks(x); b.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
    b.set_ylabel("Median abs. % error (lower = better)", fontsize=FS - 0.5); b.set_title("(b) Digital twin", fontsize=FS)
    for ax in (a, b): ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.3, w_pad=0.6); plt.savefig(out, dpi=600); plt.close()
if __name__ == "__main__":
    calibration_cis(); excluded_vs_included(); fig1(); print("done")
