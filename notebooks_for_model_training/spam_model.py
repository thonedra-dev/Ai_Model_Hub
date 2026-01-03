import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing, callbacks, optimizers, losses, metrics
import re
import pickle
import subprocess

plt.ion()
np.random.seed(42)
tf.random.set_seed(42)

print("✅ TensorFlow:", tf.__version__, "| GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
print("="*60)

# 1. LOAD DATA
print("📂 Loading dataset...")
df = pd.read_csv('datasets/spam_Emails_data.csv')
print(f"Dataset shape: {df.shape}")
print("="*60)

# 2. CLEAN DATA
print("🧹 Cleaning data...")
df = df.dropna(subset=['label'])
df['label'] = df['label'].str.lower()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'escapenumber|escapelong', ' ', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'mailto\S+|http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\?!,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_cleaned'] = df['text'].apply(clean_text)

# Filter by length (99th percentile)
df['word_count'] = df['text_cleaned'].apply(lambda x: len(x.split()))
length_99 = int(np.percentile(df['word_count'], 99))
df = df[df['word_count'] <= length_99].copy()
df = df.drop(columns=['word_count'])
print(f"Filtered dataset shape: {df.shape}")
print("="*60)

# 3. EDA
print("📊 Class distribution:")
print(df['label'].value_counts())
print("="*60)

# 4. SPLIT DATA
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label'])

X_train = train_df['text_cleaned'].values
y_train = train_df['label'].values
X_val = val_df['text_cleaned'].values
y_val = val_df['label'].values
X_test = test_df['text_cleaned'].values
y_test = test_df['label'].values

print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
print("="*60)

# 5. TOKENIZATION
print("🔠 Tokenizing text...")
max_vocab = 20000
tokenizer = preprocessing.text.Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

train_seq_lengths = [len(seq) for seq in X_train_seq]
max_seq_len = int(np.percentile(train_seq_lengths, 90))
print(f"Max sequence length (90th percentile): {max_seq_len}")

X_train_pad = preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_seq_len, padding='post', truncating='post')
X_val_pad = preprocessing.sequence.pad_sequences(X_val_seq, maxlen=max_seq_len, padding='post', truncating='post')
X_test_pad = preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_seq_len, padding='post', truncating='post')
print(f"Padded training shape: {X_train_pad.shape}")
print("="*60)

# 6. ENCODE LABELS
print("🏷️ Encoding labels...")
label_mapping = {'ham': 0, 'spam': 1}
y_train_enc = np.array([label_mapping[label] for label in y_train])
y_val_enc = np.array([label_mapping[label] for label in y_val])
y_test_enc = np.array([label_mapping[label] for label in y_test])
print("="*60)

# 7. BUILD MODEL
print("🧠 Building transformer model...")
EMBEDDING_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_DIM = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.4

def build_transformer_model(vocab_size, max_length):
    inputs = layers.Input(shape=(max_length,))
    embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM, input_length=max_length, mask_zero=True)(inputs)
    
    positions = tf.range(start=0, limit=max_length, delta=1)
    position_embedding = layers.Embedding(input_dim=max_length, output_dim=EMBEDDING_DIM)(positions)
    x = embedding + position_embedding
    
    x = layers.MultiHeadAttention(num_heads=TRANSFORMER_HEADS, key_dim=TRANSFORMER_DIM, dropout=DROPOUT_RATE)(x, x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inputs, outputs=outputs)

vocab_size = min(max_vocab, len(tokenizer.word_index))
model = build_transformer_model(vocab_size, max_seq_len)
model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=2e-4),
    loss=losses.BinaryCrossentropy(),
    metrics=['accuracy', metrics.Precision(), metrics.Recall()]
)
print("="*60)

# 8. TRAIN WITH 2-3 EPOCHS
print("🚂 Training for 3 epochs...")
training_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    callbacks.ModelCheckpoint('best_spam_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

BATCH_SIZE = 16
EPOCHS = 2  # Changed to 3 epochs as requested

history = model.fit(
    X_train_pad, y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=training_callbacks,
    verbose=1
)
print("="*60)

# 9. VALIDATION CHECK
print("🔍 Validation metrics check...")
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val_pad, y_val_enc, verbose=0)
print(f"Val Accuracy: {val_accuracy:.4f}")
print(f"Val Precision: {val_precision:.4f}")
print(f"Val Recall: {val_recall:.4f}")
print("="*60)

# 10. FINAL TEST EVALUATION
print("📈 Final test evaluation...")
model = models.load_model('best_spam_model.h5')
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_pad, y_test_enc, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

y_pred_proba = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("\n📋 Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print("="*60)

# 11. SAVE FOR PRODUCTION
print("💾 Saving production files...")
model.save('spam_classifier.h5')
print("✅ Model saved as 'spam_classifier.h5'")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as 'tokenizer.pkl'")

with open('label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)
print("✅ Label mapping saved as 'label_mapping.pkl'")

with open('requirements.txt', 'w') as f:
    subprocess.call(['pip', 'freeze'], stdout=f)
print("✅ Requirements saved as 'requirements.txt'")

print("="*60)
print(f"🎉 FINAL TEST ACCURACY: {test_accuracy:.2%}")
if test_accuracy >= 0.95:
    print("✅ Excellent! Model is ready for deployment.")
elif test_accuracy >= 0.85:
    print("⚠️  Good, but consider more training or data augmentation.")
else:
    print("❌ Needs improvement. Check data quality and model architecture.")