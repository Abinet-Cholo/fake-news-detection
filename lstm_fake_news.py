import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle  # For saving the tokenizer
import os  # For directory handling

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Get current working directory
data_dir = os.getcwd()

# Load datasets
true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))
fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))

# Add labels
true_df['label'] = 1
fake_df['label'] = 0

# Combine and shuffle data
df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

# Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    return text

df['text'] = df['text'].apply(clean_text)

# Split features and labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizing the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences to the same length
max_length = 500  # Reduced max_length to optimize memory usage
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Building the LSTM model
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate on test set
loss, accuracy = lstm_model.evaluate(X_test_pad, y_test)
print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype(int)

# Classification report
print("\nLSTM Model Performance:\n")
print(classification_report(y_test, y_pred_lstm))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lstm), annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM Confusion Matrix')
plt.show()

# Save the trained model
model_save_path = os.path.join(data_dir, 'lstm_fake_news_model.h5')
lstm_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Save the tokenizer
tokenizer_save_path = os.path.join(data_dir, 'tokenizer.pkl')
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved to {tokenizer_save_path}")
