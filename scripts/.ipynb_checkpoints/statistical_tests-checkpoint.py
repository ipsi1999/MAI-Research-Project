"""
Statistical significance testing for Part 2 evaluation results.
Compares ASSD and HD90 across k2, k3, k4 using Wilcoxon signed-rank test
with Bonferroni correction for multiple comparisons.

Usage:
    python statistical_tests.py
"""

import pandas as pd
import numpy as np
from scipy import stats

EVAL_DIR = "/nesi/project/uoa04396/isin038/results/part2/evaluation"
ALPHA = 0.05
N_COMPARISONS = 3  # k2vsk3, k3vsk4, k2vsk4
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS

# Load results
k2 = pd.read_csv(f"{EVAL_DIR}/results_k2.csv")
k3 = pd.read_csv(f"{EVAL_DIR}/results_k3.csv")
k4 = pd.read_csv(f"{EVAL_DIR}/results_k4.csv")

# Merge on subject to ensure paired data
merged = k2[["subject", "assd_mm", "hd90_mm"]].rename(
    columns={"assd_mm": "assd_k2", "hd90_mm": "hd90_k2"}
).merge(
    k3[["subject", "assd_mm", "hd90_mm"]].rename(
        columns={"assd_mm": "assd_k3", "hd90_mm": "hd90_k3"}
    ), on="subject"
).merge(
    k4[["subject", "assd_mm", "hd90_mm"]].rename(
        columns={"assd_mm": "assd_k4", "hd90_mm": "hd90_k4"}
    ), on="subject"
)

print(f"Paired subjects: {len(merged)}")
print(f"Significance level: {ALPHA}")
print(f"Bonferroni-corrected threshold: {BONFERRONI_ALPHA:.4f}")
print()

# Summary stats
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
for res in ["k2", "k3", "k4"]:
    assd = merged[f"assd_{res}"]
    hd90 = merged[f"hd90_{res}"]
    print(f"\n{res}:")
    print(f"  ASSD  mean={assd.mean():.4f} ± {assd.std():.4f}  median={assd.median():.4f}")
    print(f"  HD90  mean={hd90.mean():.4f} ± {hd90.std():.4f}  median={hd90.median():.4f}")

# Wilcoxon signed-rank tests
comparisons = [
    ("k2", "k3"),
    ("k3", "k4"),
    ("k2", "k4"),
]

print()
print("=" * 60)
print("WILCOXON SIGNED-RANK TESTS (paired, two-sided)")
print(f"Bonferroni correction: p < {BONFERRONI_ALPHA:.4f} for significance")
print("=" * 60)

for metric in ["assd", "hd90"]:
    print(f"\n--- {metric.upper()} ---")
    for res_a, res_b in comparisons:
        col_a = f"{metric}_{res_a}"
        col_b = f"{metric}_{res_b}"
        vals_a = merged[col_a].values
        vals_b = merged[col_b].values

        # Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(vals_a, vals_b)

        # Effect size: r = Z / sqrt(N)
        # Approximate Z from the statistic
        n = len(vals_a)
        z_score = stats.norm.ppf(p_value / 2)
        effect_size = abs(z_score) / np.sqrt(n)

        significant = p_value < BONFERRONI_ALPHA
        sig_str = "YES ***" if significant else "no"

        mean_diff = vals_a.mean() - vals_b.mean()
        direction = f"{res_a} > {res_b}" if mean_diff > 0 else f"{res_a} < {res_b}"

        print(f"\n  {res_a} vs {res_b}:")
        print(f"    {res_a} mean: {vals_a.mean():.4f}   {res_b} mean: {vals_b.mean():.4f}")
        print(f"    Mean difference: {mean_diff:.4f} ({direction})")
        print(f"    W statistic: {stat:.1f}")
        print(f"    p-value: {p_value:.6f}")
        print(f"    Effect size (r): {effect_size:.4f}")
        print(f"    Significant (p < {BONFERRONI_ALPHA:.4f}): {sig_str}")

print()
print("=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("Effect size (r): small=0.1, medium=0.3, large=0.5")
print(f"Bonferroni correction applied: {N_COMPARISONS} comparisons, alpha={ALPHA} -> {BONFERRONI_ALPHA:.4f}")
print("Positive mean difference = first resolution has HIGHER (worse) error")
