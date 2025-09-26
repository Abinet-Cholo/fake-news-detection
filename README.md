# Fake News Detection

📌 **Project Overview**  
This project implements **Fake News Detection** using two different approaches:
- Logistic Regression with TF-IDF vectorization  
- LSTM Neural Network  

Both models are trained on the Kaggle Fake News dataset and evaluated to compare their performance. Unlike many toy examples that overfit easily, this project focuses on building models that generalize well, keeping results realistic and reproducible.

---

## 📂 Dataset
We use a dataset containing real and fake news articles. After preprocessing, the dataset is split into training and testing sets.

- **True.csv** → Legitimate news articles  
- **Fake.csv** → Fake news articles  
- **Labels** → `1 = Real news`, `0 = Fake news`  

---

## 🛠 Models & Training

### 1️⃣ Logistic Regression with TF-IDF
- Used **TF-IDF vectorization** to convert text into numerical features.  
- Trained using **Scikit-learn’s Logistic Regression**.  
- **Achieved ~83% accuracy and F1-score** on test data, showing strong performance on smaller datasets due to TF-IDF capturing important word-level features.

📊 *Evaluation Metrics* (approximate):
```
precision    recall  f1-score   support
0           ~0.83     ~0.82     ~0.82    4600+
1           ~0.84     ~0.85     ~0.84    4300+
accuracy                          0.83    8900+
```

---

### 2️⃣ LSTM Neural Network
- Built with **Keras Tokenizer** and **LSTM layers** to capture sequential patterns in text.  
- Trained on a GPU for faster performance.  
- **Achieved ~79–81% accuracy and F1-score** on test data, slightly underperforming Logistic Regression. This is expected, as deep learning models often require **larger datasets** and careful hyperparameter tuning.

📊 *Evaluation Metrics* (approximate):
```
precision    recall  f1-score   support
0           ~0.80     ~0.77     ~0.78    4600+
1           ~0.78     ~0.81     ~0.79    4300+
accuracy                          0.80    8900+
```

---

## 🚀 How to Use
```bash
# Clone this repository
git clone https://github.com/Abinet-Cholo/fake-news-detection.git
cd fake-news-detection

# Install required libraries
pip install -r requirements.txt

# Run Logistic Regression Model
python logistic_regression_fake_news.py

# Run LSTM Model
python lstm_fake_news.py
```

---

## 📊 Model Comparison

| Model               | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | ~83%     | ~83%      | ~84%   | ~83%     |
| LSTM                 | ~80%     | ~80%      | ~79%   | ~79%     |

✔️ **Key Insight**: Logistic Regression outperformed LSTM on this dataset because TF-IDF with a linear model is more effective for smaller, structured datasets. LSTMs can capture sequence dependencies but typically need **larger, more diverse corpora** and extensive tuning to shine.

---

## 📢 Credits
- Dataset: Kaggle Fake News Dataset  
- Implemented by **Abinet Bushura Cholo**  
- If you found this helpful, consider ⭐ starring this repo!
