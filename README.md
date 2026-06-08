# CNN HPO for Agricultural Pest Classification
 
Source code and data for a bachelor thesis comparing Random Search (RS) and Bayesian Optimization (BO)
for hyperparameter optimization when fine-tuning MobileNetV2 on three agricultural pest datasets:
ASDID (soybean diseases), DeepWeeds (weeds), and IP41 (crop pests).
 
## Setup
 
A CUDA-capable GPU is recommended, as training the full set of models is computationally expensive.
 
Install the dependencies in a virtual Python environment, preferably in Ubuntu:
 
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```
 
## Scripts
 
Fill in the variables at the top of each script before running. `dataset_name` must be the same in all
four files. `dataset_path` points to a folder with `train`/`val`/`test` subfolders organized by class.
Run the scripts in this order:
 
1. **`hyperparameter_optimization.py`** — RS + BO Optuna studies for one seed (run once per seed).
   - Fill in: `seed` (a single int), `dataset_path`, `number_of_classes`, `dataset_name`.
2. **`train_and_test_models.py`** — fine-tunes a model from each best HPC, saves test scores.
   - Fill in: `seeds` (a list of ints), `dataset_path`, `number_of_classes`, `dataset_name`.
3. **`wilcoxon_tests.py`** / **`mcnemar_tests.py`** — statistical comparison of RS vs BO.
   - `wilcoxon_tests.py` — fill in: `dataset_name`.
   - `mcnemar_tests.py` — fill in: `seeds` (a list of ints), `dataset_path`, `dataset_name`.

Seeds used in the thesis: `42, 1337, 2024, 9001, 31415, 771294, 5921, 813745, 16652, 1086`.
 
Note: These seeds control sampler initialization, shuffling, and augmentation, but nondeterministic GPU
scheduling means exact results may differ slightly from the thesis.
 
## Datasets
 
The datasets are available from their original authors and were used under their respective terms of use.
 
**ASDID** (Auburn Soybean Disease Image Dataset) — Bevers, N., Sikora, E. J., & Hardy, N. B. (2022).
Soybean Disease Identification Using Original Field Images and Transfer Learning with Convolutional
Neural Networks. *Computers and Electronics in Agriculture*, 203, Article 107449.
DOI: 10.1016/j.compag.2022.107449.
 
**DeepWeeds** — Olsen, A. et al. (2019). DeepWeeds: A Multiclass Weed Species Image Dataset for Deep
Learning. *Scientific Reports*, 9, Article 2058. DOI: 10.1038/s41598-018-38343-3.
 
**IP41** — Wang, K., Chen, K., Du, H., Liu, S., Xu, J., Zhao, J., Chen, H., Liu, Y., & Liu, Y. (2022).
New Image Dataset and New Negative Sample Judgment Method for Crop Pest Recognition Based on Deep
Learning Models. *Ecological Informatics*, 69, Article 101620. DOI: 10.1016/j.ecoinf.2022.101620.
