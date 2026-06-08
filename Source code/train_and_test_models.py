import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import gc
import yaml
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU online?
print("GPUs:", tf.config.list_physical_devices("GPU"))

seeds = [] # enter array of seeds here

def set_reproducibility_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

# Training configuration and dataset settings
number_of_epochs = 30
early_stopping_patience = 8
dataset_path = None # replace this with dataset path
number_of_classes = None # replace this with number of classes in dataset
dataset_name = "" # enter dataset name here

# Loading and preprocessing of the dataset
def make_dataset(split, seed):

    augment = (split == "train")

    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_path, split),
        image_size = (224, 224),
        batch_size = 32,
        shuffle = (split == "train"),
        seed = seed,
        label_mode = "categorical",
    )

    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal", seed = seed),
            layers.RandomRotation(0.1, seed = seed),
            layers.RandomZoom(0.1, seed = seed),
        ])
        dataset = dataset.map(
            lambda x, y: (aug(x, training = True), y),
            num_parallel_calls = tf.data.AUTOTUNE
        )

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    dataset = dataset.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls = tf.data.AUTOTUNE
    )

    return dataset.prefetch(1)

# Model building
def build_model(learning_rate, dropout_rate, freezing_ratio, num_dense_layers, dense_layer_size):
    
    backbone_model = MobileNetV2(
        input_shape = (224, 224, 3),
        include_top = False,
        weights = "imagenet",
        pooling = "avg",
    )

    n_layers = len(backbone_model.layers)
    freeze_up_to = int(n_layers * freezing_ratio)

    for i, layer in enumerate(backbone_model.layers):
        layer.trainable = (i >= freeze_up_to)

    x = backbone_model.output

    for _ in range(num_dense_layers):
        x = layers.Dense(dense_layer_size, activation = "relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(number_of_classes, activation = "softmax")(x)

    model = models.Model(
        inputs = backbone_model.input,
        outputs = outputs
    )

    model.compile(
        optimizer = optimizers.Adam(learning_rate = learning_rate),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model

# Wrapper subprocess
def subprocess_wrapper(queue, hyperparameters, save_path, seed):
    
    set_reproducibility_seed(seed)

    train_split = make_dataset("train", seed)
    validation_split = make_dataset("val", seed)
    test_split = make_dataset("test", seed)

    model = build_model(**hyperparameters)

    early_stopping = EarlyStopping(
        monitor = "val_loss",
        patience = early_stopping_patience,
        restore_best_weights = True,
    ) 

    model.fit(
        train_split,
        validation_data = validation_split,
        epochs = number_of_epochs,
        callbacks = [early_stopping],
        verbose = 1,
    )

    model.save(save_path)
    print(f"Model saved to {save_path}")

    actual_labels, predicted_labels = [], []

    for images, labels in test_split:
        predictions = model.predict(images, verbose = 0)
        actual_labels.extend(np.argmax(labels.numpy(), axis = 1))
        predicted_labels.extend(np.argmax(predictions, axis = 1))

    results = {
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "Precision": precision_score(
            actual_labels,
            predicted_labels,
            average = "macro",
            zero_division = 0
        ),
        "Recall": recall_score(
            actual_labels,
            predicted_labels,
            average = "macro",
            zero_division = 0
        ),
        "F1-score": f1_score(
            actual_labels,
            predicted_labels,
            average = "macro",
            zero_division = 0
        ),
    }
    
    queue.put(results)

    tf.keras.backend.clear_session()
    gc.collect()

# Main
if __name__ == "__main__":
    mp.set_start_method("spawn", force = True)

    samplers = ["RandomSampler", "TPESampler"]

    all_results = []

    for seed in seeds:
        for sampler_name in samplers:
            folder = os.path.join(dataset_name, str(seed), sampler_name)

            yaml_path = os.path.join(
                folder,
                f"{sampler_name}_{seed}.yaml"
            )

            model_save_path = os.path.join(
                folder,
                f"{sampler_name}_{seed}.keras"
            )

            print(f"\n--- Training {sampler_name} | Seed {seed} ---")
            print(f"Loading hyperparameters from: {yaml_path}")

            with open(yaml_path, "r") as f:
                hyperparameters = yaml.safe_load(f)

            ctx = mp.get_context("spawn")
            queue = ctx.Queue()

            p = ctx.Process(
                target = subprocess_wrapper,
                args = (queue, hyperparameters, model_save_path, seed)
            )

            p.start()
            p.join()

            results = queue.get()
            results["Seed"] = seed
            results["Sampler"] = sampler_name

            all_results.append(results)

            print(f"Results for {sampler_name} | Seed {seed}: {results}")

    df = pd.DataFrame(all_results)
    df.to_csv(f"{dataset_name}/final_test_results.csv", index = False)

    print("Complete!")