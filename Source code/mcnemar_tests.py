import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.stats.contingency_tables import mcnemar

dataset_name = "" # enter dataset name here

seeds = [] # enter array of seeds for model pairs to test
dataset_path = None # replace this with dataset path
test_split = "test"

def make_dataset(split: str) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, split),
        image_size = (224, 224), batch_size = 32,
        shuffle = False, label_mode = "categorical",
    )
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls = tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def get_predictions(model_path: str, dataset: tf.data.Dataset):
    model = tf.keras.models.load_model(model_path)
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose = 0)
        y_true.extend(np.argmax(labels.numpy(), axis = 1))
        y_pred.extend(np.argmax(probs, axis = 1))
    tf.keras.backend.clear_session()
    gc.collect()
    return np.array(y_true), np.array(y_pred)

def build_mcnemar_table(y_true, y_pred_rs, y_pred_bo):
    rs_correct = y_pred_rs == y_true
    bo_correct = y_pred_bo == y_true
    table = np.array([
        [np.sum( rs_correct &  bo_correct), np.sum( rs_correct & ~bo_correct)],
        [np.sum(~rs_correct &  bo_correct), np.sum(~rs_correct & ~bo_correct)],
    ])
    return table, table[0, 1], table[1, 0]

test_ds = make_dataset(test_split)

mcnemar_n01 = []
mcnemar_n10 = []
mcnemar_pvals = []

for seed in seeds:
    y_true_rs, y_pred_rs = get_predictions(f"{dataset_name}/{seed}/RandomSampler/RandomSampler_{seed}.keras", test_ds)
    y_true_bo, y_pred_bo = get_predictions(f"{dataset_name}/{seed}/TPESampler/TPESampler_{seed}.keras", test_ds)

    table, b, c = build_mcnemar_table(y_true_rs, y_pred_rs, y_pred_bo)
    result = mcnemar(table, exact = True)
    (n00, n01), (n10, n11) = table

    mcnemar_n01.append(int(n01))
    mcnemar_n10.append(int(n10))
    mcnemar_pvals.append(result.pvalue)

    print(f"\nSeed {seed}")
    print(f"  n01 (RS correct, BO wrong) = {n01}")
    print(f"  n10 (RS wrong, BO correct) = {n10}")
    print(f"  p-value = {result.pvalue:.6f} = {'significant' if result.pvalue < 0.05 else 'not significant'}")

rows = ""
for i, seed in enumerate(seeds):
    sig = "\\textbf{Yes}" if mcnemar_pvals[i] < 0.05 else "No"
    rows += f"{seed} & {mcnemar_n01[i]} & {mcnemar_n10[i]} & {mcnemar_pvals[i]:.5f} & {sig} \\\\\n"

print(f"{rows}")

results_df = pd.DataFrame({
    "Seed": seeds,
    "n01_RS_correct_BO_wrong": mcnemar_n01,
    "n10_BO_correct_RS_wrong": mcnemar_n10,
    "p_value": mcnemar_pvals,
    "Significant": [p < 0.05 for p in mcnemar_pvals],
})

out_path = f"{dataset_name}/{dataset_name}_mcnemar_results.csv"
results_df.to_csv(out_path, index=False)

print(f"Saved results: {out_path}")