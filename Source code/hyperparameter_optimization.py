import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import gc
import yaml
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPU online?
print("GPUs:", tf.config.list_physical_devices("GPU"))

seed = None # Enter the shared seed to pair studies on
def set_reproducibility_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
set_reproducibility_seed(seed)

# Training configuration and dataset settings
number_of_trials = 50
number_of_epochs = 30
early_stopping_patience = 8
dataset_path = None # replace this with dataset path
number_of_classes = None # replace this with number of classes in dataset
dataset_name = "" # enter dataset name here

# Loading and preprocessing of dataset
def make_dataset(split):

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
        dataset = dataset.map(lambda x, y: (aug(x, training = True), y), num_parallel_calls = tf.data.AUTOTUNE)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    dataset = dataset.map(lambda x, y: (preprocess(x), y), num_parallel_calls = tf.data.AUTOTUNE)
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

    model = models.Model(inputs = backbone_model.input, outputs = outputs)

    model.compile(
        optimizer = optimizers.Adam(learning_rate = learning_rate), 
        loss = "categorical_crossentropy", 
        metrics = ["accuracy"]
    )

    return model

# Wrapper subprocess
def subprocess_wrapper(queue, hyperparameters):
    
    set_reproducibility_seed(seed)

    train_split = make_dataset("train")
    validation_split = make_dataset("val")

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

    actual_labels, predicted_labels = [], []
    for images, labels in validation_split:
        predictions = model.predict(images, verbose = 0)
        actual_labels.extend(np.argmax(labels.numpy(), axis = 1))
        predicted_labels.extend(np.argmax(predictions, axis = 1))

    results = {
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "Precision": precision_score(actual_labels, predicted_labels, average = "macro", zero_division = 0),
        "Recall": recall_score(actual_labels, predicted_labels, average = "macro", zero_division = 0),
        "F1-score": f1_score(actual_labels, predicted_labels, average = "macro", zero_division = 0),
    }
    
    queue.put(results)
    tf.keras.backend.clear_session()
    gc.collect()

# Optuna objective
def hyperparameter_optimization(trial):
    
    # Hyperparameters and their search space
    hyperparameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4),
        "freezing_ratio": trial.suggest_float("freezing_ratio", 0.7, 0.9),
        "num_dense_layers": trial.suggest_int("num_dense_layers", 1, 3),
        "dense_layer_size": trial.suggest_categorical("dense_layer_size", [128, 256, 512])
    }

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target = subprocess_wrapper, args = (queue, hyperparameters))
    p.start()
    p.join() 

    res = queue.get()
    
    for key, value in res.items():
        trial.set_user_attr(key, value)
    
    return res["F1-score"]

# Optuna Study
def run_study(study_name, sampler, algorithm_name, seed_directory):

    sampler_directory = os.path.join(seed_directory, algorithm_name)
    os.makedirs(sampler_directory)

    yaml_path = os.path.join(
        sampler_directory,
        f"{algorithm_name}_{seed}.yaml"
    )

    database_path = os.path.join(
        seed_directory,
        f"{seed}.db"
    )

    study = optuna.create_study(
        direction = "maximize",
        sampler = sampler,
        study_name = study_name,
        storage = f"sqlite:///{database_path}",
        load_if_exists = True,
        pruner = optuna.pruners.NopPruner(),
    )

    study.optimize(
        hyperparameter_optimization,
        n_trials = number_of_trials,
        show_progress_bar = True,
        gc_after_trial = True,
    )

    with open(yaml_path, "w") as f:
        yaml.dump(study.best_params, f, default_flow_style = False)

    return study

# Main
if __name__ == "__main__":
    mp.set_start_method("spawn", force = True)

    seed_directory =  os.path.join(dataset_name, str(seed))
    os.makedirs(seed_directory, exist_ok=True)       

    rs_study = run_study(
        "rs_study",
        RandomSampler(seed = seed),
        "RandomSampler",
        seed_directory,
    )

    bo_study = run_study(
        "bo_study",
        TPESampler(seed = seed, n_startup_trials = 5),
        "TPESampler",
        seed_directory,
    )

    print("Complete!")