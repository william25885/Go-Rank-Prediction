# Go Rank Prediction System

## Project Overview
This is a deep-learning-based Go (Weiqi) rank prediction system that uses model ensembling to predict a player’s rank from **1D to 9D**.

---

## File Descriptions

### 1. `train.py` — Model Training Script
**Purpose:** Train multiple deep learning models and perform model ensembling.

**Key Features:**
- Supports two sequence models: **Transformer** and **BiLSTM**
- Uses **5-fold cross-validation** for evaluation
- Integrates **tabular features** and **meta-learning**
- Supports **GPU-accelerated** training
- Automatically saves trained models

**Usage:**
```bash
python train.py --train_dir train_set --out_dir . --gpu --seq_len 120 --epochs 30
```

**Arguments:**
- `--train_dir`: Training data directory (default: `train_set`)
- `--out_dir`: Output directory (default: current directory)
- `--gpu`: Enable GPU acceleration
- `--seq_len`: Sequence length (default: 120)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.0003)
- `--seed`: Random seed (default: 42)

---

### 2. `Q5.py` — Prediction Script
**Purpose:** Use trained models to predict ranks for the test set.

**Key Features:**
- Loads a pretrained stacked/ensemble model
- Supports multiple ensembling strategies
- Automatically processes test files and generates predictions
- Outputs a CSV submission file

**Usage:**
```bash
python Q5.py --test_dir test_set --model_path model_stackx.pkl --out_csv submission.csv
```

**Arguments:**
- `--test_dir`: Test data directory (default: `test_set`)
- `--model_path`: Path to model file (default: `model_stackx.pkl`)
- `--out_csv`: Output CSV file (default: `submission.csv`)

---

## Data Format

### Input Files
- **Training data:** `log_XD_policy_train.txt` (X = 1–9, representing the rank)
- **Test data:** `X.txt` (X = file index)

### Content Structure
Each line contains:
- Game ID: `Game X:`
- Move: `B[coord]` or `W[coord]`
- Feature vectors:
  - Policy vector (9D)
  - Value vector (9D)
  - Rank probability vector (9D)
  - Strength (1D)
  - Winrate, Lead, Uncertainty (3D)

---

## Model Architecture

### 1. Sequence Models
- **TinyTransformer**: Transformer-based sequence model
- **BiLSTM**: Bidirectional LSTM model
- Feature dimension: **79**
- Sequence length: **120** moves

### 2. Tabular Models
- **CatBoost** or **HistGradientBoostingClassifier**
- Extracts statistical features from each game
- Includes features from opening, midgame, and endgame

### 3. Meta-learning
- Uses **Logistic Regression** for model fusion
- Combines sequence models, tabular models, and side features

---

## Feature Engineering

### Sequence Features (79D)
- Base: Policy, Value, RankP (each 9D) + Strength, Winrate, Lead, Uncertainty
- Derived: max values, entropies (each 6D)
- Differences: first-order differences (37D), second-order differences (3D)
- Position: color, normalized position

### Tabular Features
- Statistics: mean, std, min, max, median, quartiles, skewness
- Segments: opening/midgame/endgame features
- Preferences: entropy, max, mode

### Side Features (10D)
- Policy/Value/RankP entropies
- Log game length
- Winrate standard deviation
- Mean absolute Lead
- Mean Uncertainty
- Opening/midgame/endgame Policy entropy

---

## Training Pipeline
1. Data parsing: extract game data from logs
2. Feature engineering: compute sequence and statistical features
3. Model training:
   - Sequence models (Transformer + BiLSTM)
   - Tabular models (CatBoost/HGBT)
4. Ensembling: meta-learning to combine all models
5. Saving: save the complete model stack

---

## Inference Pipeline
1. Load model: load the model stack from a pickle file
2. Data processing: parse test files and extract features
3. Multi-model prediction:
   - Sequence predictions (multi-view)
   - Tabular predictions
   - Side feature computation
4. Ensembling: meta-learner combines predictions
5. Output: generate a CSV submission file

---

## Dependencies
```
torch
numpy
pandas
scikit-learn
catboost (optional)
scipy
pathlib
concurrent.futures
json
pickle
argparse
math
random
warnings
```

---

## Outputs

### Training Stage
- `model_stackx.pkl`: Complete stacked model
- `train_summary.json`: Training summary information

### Prediction Stage
- `submission.csv`: Prediction results (includes `id` and `rank` columns)

---

## Notes
1. **GPU support:** GPU training is recommended for speed.
2. **Memory:** Sequence models require more memory; at least **8GB RAM** is recommended.
3. **Training time:** Full training takes about **2–4 hours**, depending on hardware.
4. **Model size:** The full model file is about **100–200MB**.

---

## Performance Metrics
- **Sequence model accuracy:** Transformer ~38.5%, BiLSTM ~36.5%
- **Tabular model accuracy:** ~20.3%
- **Ensembled model accuracy:** ~42.8%

---

## Troubleshooting
1. **Out of memory:** reduce `batch_size` or `seq_len`
2. **GPU errors:** check CUDA compatibility
3. **File read errors:** verify data paths and file formats
4. **Model loading failure:** ensure the pickle file is complete

---

## Version Info
- **Version:** 5.0
- **Last updated:** 2024
- **Compatibility:** Python 3.8+, PyTorch 1.12+


