import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle  # For saving the model and vectorizer
import os  # For directory handling

# Get current working directory
data_dir = os.getcwd()

# Load datasets
true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))
fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))

# Add labels
true_df['label'] = 1  # Real news
fake_df['label'] = 0  # Fake news

# Combine datasets
df = pd.concat([true_df, fake_df]).reset_index(drop=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Basic text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([c for c in text if c.isalpha() or c == ' '])  # Remove punctuation and numbers
    return text

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Split features and labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_tfidf)

# Classification report
print("Logistic Regression Performance:\n")
print(classification_report(y_test, y_pred_lr))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Save the trained model
model_save_path = os.path.join(data_dir, 'logistic_regression_model.pkl')
with open(model_save_path, 'wb') as model_file:
    pickle.dump(lr_model, model_file)
print(f"Model saved to {model_save_path}")

# Save the TF-IDF vectorizer
vectorizer_save_path = os.path.join(data_dir, 'tfidf_vectorizer.pkl')
with open(vectorizer_save_path, 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
print(f"TF-IDF Vectorizer saved to {vectorizer_save_path}")
