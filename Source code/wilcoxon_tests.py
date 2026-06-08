from scipy.stats import wilcoxon
import pandas as pd

dataset_name = "" # enter dataset name here

results_csv = f"{dataset_name}/final_test_results.csv"

df = pd.read_csv(results_csv)
rs = df[df["Sampler"] == "RandomSampler"].sort_values("Seed")
bo = df[df["Sampler"] == "TPESampler"].sort_values("Seed")

results = {
    "Accuracy":         wilcoxon(bo["Accuracy"],  rs["Accuracy"],  alternative = "two-sided", zero_method = "pratt"),
    "Precision":        wilcoxon(bo["Precision"], rs["Precision"], alternative = "two-sided", zero_method = "pratt"),
    "Recall":           wilcoxon(bo["Recall"],    rs["Recall"],    alternative = "two-sided", zero_method = "pratt"),
    "F1-score":         wilcoxon(bo["F1-score"],  rs["F1-score"],  alternative = "two-sided", zero_method = "pratt"),
}

labels = {
    "Accuracy":  "Accuracy",
    "Precision": "Precision",
    "Recall":    "Recall",
    "F1-score":  r"F\textsubscript{1}-score",
}

rows = ""
for metric, res in results.items():
    sig = "Yes" if res.pvalue < 0.05 else "No"
    rows += f"""
{labels[metric]}
& ${res.statistic}$
& ${res.pvalue}$
& {sig} \\\\
"""

print(f"{rows}")

results_df = pd.DataFrame([
    {
        "Metric": metric,
        "Statistic": res.statistic,
        "P-value": res.pvalue,
        "Significant": "Yes" if res.pvalue < 0.05 else "No"
    }
    for metric, res in results.items()
])

out_path = f"{dataset_name}/{dataset_name}_wilcoxon_results.csv"
results_df.to_csv(out_path, index = False)

print(f"Saved results: {out_path}")