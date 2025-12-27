# 圍棋段位預測系統

## 專案概述
這是一個基於深度學習的圍棋段位預測系統，使用多種模型融合技術來預測圍棋玩家的段位（1D-9D）。

## 檔案說明

### 1. train.py - 模型訓練腳本
**功能：** 訓練多個深度學習模型並進行模型融合

**主要特點：**
- 支援 Transformer 和 BiLSTM 兩種序列模型
- 使用 5-fold 交叉驗證進行模型評估
- 整合 Tabular 特徵和 Meta-learning
- 支援 GPU 加速訓練
- 自動儲存訓練好的模型

**使用方法：**
```bash
python train.py --train_dir train_set --out_dir . --gpu --seq_len 120 --epochs 30
```

**參數說明：**
- `--train_dir`: 訓練資料目錄（預設：train_set）
- `--out_dir`: 輸出目錄（預設：當前目錄）
- `--gpu`: 使用 GPU 加速
- `--seq_len`: 序列長度（預設：120）
- `--epochs`: 訓練輪數（預設：30）
- `--batch_size`: 批次大小（預設：128）
- `--lr`: 學習率（預設：0.0003）
- `--seed`: 隨機種子（預設：42）

### 2. Q5.py - 預測腳本
**功能：** 使用訓練好的模型對測試資料進行段位預測

**主要特點：**
- 載入預訓練的模型堆疊
- 支援多種模型融合策略
- 自動處理測試檔案並生成預測結果
- 輸出 CSV 格式的提交檔案

**使用方法：**
```bash
python Q5.py --test_dir test_set --model_path model_stackx.pkl --out_csv submission.csv
```

**參數說明：**
- `--test_dir`: 測試資料目錄（預設：test_set）
- `--model_path`: 模型檔案路徑（預設：model_stackx.pkl）
- `--out_csv`: 輸出 CSV 檔案（預設：submission.csv）

## 資料格式

### 輸入資料格式
- **訓練資料：** `log_XD_policy_train.txt` (X = 1-9，代表段位)
- **測試資料：** `X.txt` (X = 檔案編號)

### 資料內容
每行包含：
- 遊戲編號：`Game X:`
- 棋步：`B[座標]` 或 `W[座標]`
- 特徵向量：
  - Policy 向量（9維）
  - Value 向量（9維）
  - Rank probability 向量（9維）
  - Strength（1維）
  - Winrate, Lead, Uncertainty（3維）

## 模型架構

### 1. 序列模型
- **TinyTransformer**: 基於 Transformer 的序列模型
- **BiLSTM**: 雙向 LSTM 模型
- 特徵維度：79 維
- 序列長度：120 步

### 2. Tabular 模型
- **CatBoost** 或 **HistGradientBoostingClassifier**
- 從遊戲中提取統計特徵
- 包含開局、中局、終局的特徵

### 3. Meta-learning
- 使用 Logistic Regression 進行模型融合
- 結合序列模型、Tabular 模型和側邊特徵

## 特徵工程

### 序列特徵（79維）
- 基礎特徵：Policy, Value, RankP (各9維) + Strength, Winrate, Lead, Uncertainty
- 衍生特徵：最大值、熵值（各6維）
- 差分特徵：一階差分（37維）、二階差分（3維）
- 位置特徵：顏色、正規化位置

### Tabular 特徵
- 統計特徵：均值、標準差、最小值、最大值、中位數、四分位數、偏度
- 分段特徵：開局、中局、終局的特徵
- 偏好特徵：熵值、最大值、眾數

### 側邊特徵（10維）
- Policy/Value/RankP 熵值
- 遊戲長度對數
- Winrate 標準差
- Lead 絕對值均值
- Uncertainty 均值
- 開局/中局/終局 Policy 熵值

## 訓練流程

1. **資料解析**：從日誌檔案中提取遊戲資料
2. **特徵工程**：計算序列特徵和統計特徵
3. **模型訓練**：
   - 序列模型（Transformer + BiLSTM）
   - Tabular 模型（CatBoost/HGBT）
4. **模型融合**：使用 Meta-learning 整合所有模型
5. **模型儲存**：儲存完整的模型堆疊

## 預測流程

1. **載入模型**：從 pickle 檔案載入訓練好的模型
2. **資料處理**：解析測試檔案並提取特徵
3. **多模型預測**：
   - 序列模型預測（多個視角）
   - Tabular 模型預測
   - 側邊特徵計算
4. **模型融合**：使用 Meta-learner 整合預測結果
5. **結果輸出**：生成 CSV 提交檔案

## 依賴套件

```
torch
numpy
pandas
scikit-learn
catboost (可選)
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

## 輸出檔案

### 訓練階段
- `model_stackx.pkl`: 完整的模型堆疊
- `train_summary.json`: 訓練摘要資訊

### 預測階段
- `submission.csv`: 預測結果（包含 id 和 rank 欄位）

## 注意事項

1. **GPU 支援**：建議使用 GPU 進行訓練以提升速度
2. **記憶體需求**：序列模型需要較多記憶體，建議至少 8GB RAM
3. **訓練時間**：完整訓練約需 2-4 小時（取決於硬體配置）
4. **模型大小**：完整模型檔案約 100-200MB

## 效能指標

- **序列模型準確率**：Transformer 約 38.5%，BiLSTM 約 36.5%
- **Tabular 模型準確率**：約 20.3%
- **融合模型準確率**：約 42.8%

## 故障排除

1. **記憶體不足**：減少 batch_size 或 seq_len
2. **GPU 錯誤**：檢查 CUDA 版本相容性
3. **檔案讀取錯誤**：確認資料路徑和檔案格式正確
4. **模型載入失敗**：檢查 pickle 檔案是否完整

## 版本資訊

- **版本**：5.0
- **最後更新**：2024年
- **相容性**：Python 3.8+, PyTorch 1.12+
