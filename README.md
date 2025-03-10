# Fake News Detection

## 📌 Project Overview
This project implements **Fake News Detection** using two different models:
1. **Logistic Regression with TF-IDF Vectorization**
2. **LSTM Neural Network**

Both models are trained on a Kaggle dataset for fake news detection and evaluated to compare their performance.

---

## 📂 Dataset
We use a dataset containing real and fake news articles. The dataset is preprocessed and split into training and testing sets before feeding into the models.

- **True.csv** → Contains legitimate news articles.
- **Fake.csv** → Contains fake news articles.
- Labels:
  - `1` → Real news
  - `0` → Fake news

---

## 🛠 Models & Training
### 1️⃣ **Logistic Regression (TF-IDF)**
- Uses **TF-IDF vectorization** to convert text into numerical features.
- Trained using **Scikit-learn's Logistic Regression**.
- Achieved **99% accuracy** on test data.

#### **Evaluation Metrics**
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4642
           1       0.99      0.99      0.99      4338

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980
```

### 2️⃣ **LSTM (Neural Network)**
- Uses **Keras Tokenizer & LSTM layers** to process text data.
- Trained on a **GPU** for better performance.
- Achieved **91.29% accuracy** on test data.

#### **Training Progress**
```
Epoch 1: accuracy=66.5%, val_accuracy=92.26%
Epoch 2: accuracy=88.54%, val_accuracy=93.01%
Epoch 3: accuracy=92.65%, val_accuracy=93.35%
Epoch 4: accuracy=88.79%, val_accuracy=84.38%
Epoch 5: accuracy=89.79%, val_accuracy=91.34%
```

#### **Evaluation Metrics**
```
              precision    recall  f1-score   support

           0       0.98      0.85      0.91      4633
           1       0.86      0.98      0.92      4347

    accuracy                           0.91      8980
   macro avg       0.92      0.92      0.91      8980
weighted avg       0.92      0.91      0.91      8980
```

---

## 🚀 How to Use
1. **Clone this repository:**
   ```bash
   git clone https://github.com/Abinet-Cholo/fake-news-detection.git
   cd fake-news-detection
   ```
2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Logistic Regression Model:**
   ```bash
   python logistic_regression_fake_news.py
   ```
4. **Run LSTM Model:**
   ```bash
   python lstm_fake_news.py
   ```

---

## 📊 Model Comparison
| Model | Accuracy | Precision | Recall | F1-score |
|--------|----------|------------|--------|---------|
| **Logistic Regression** | 99% | 99% | 99% | 99% |
| **LSTM** | 91.29% | 92% | 92% | 91% |

- **Logistic Regression** performed **better** overall, achieving near-perfect accuracy.
- **LSTM** is still effective but might require more tuning (e.g., hyperparameter tuning, more epochs, or different architectures).

---

## 📢 Credits
- Dataset: [Kaggle Fake News Dataset](https://www.kaggle.com/competitions/fake-news)
- Implemented by **Abinet Bushura Cholo**
- If you found this helpful, consider ⭐ **starring this repo!**

---

