# -*- coding: utf-8 -*-
"""
LAB 6 - Intent Classification (Version cải thiện)
------------------------------------------------
Bao gồm 5 nhiệm vụ:
1. TF-IDF + Logistic Regression
2. Word2Vec trung bình + Dense Layer
3. LSTM với Embedding Word2Vec pretrained
4. LSTM học Embedding từ đầu
5. Đánh giá, So sánh và Phân tích (đã tối ưu huấn luyện)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Tăng tốc khởi động TensorFlow

import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Đặt random seed để tái lập kết quả
tf.random.set_seed(42)
np.random.seed(42)


# =====================================================
# BƯỚC 0: TẢI DỮ LIỆU VÀ MÃ HÓA NHÃN
# =====================================================
def load_and_prepare_data():
    base_path = 'data/hwu/hwu'
    train_path = os.path.join(base_path, 'train.csv')
    val_path = os.path.join(base_path, 'val.csv')
    test_path = os.path.join(base_path, 'test.csv')

    def read_file(path):
        for sep in ['\t', ',', ';', '|']:
            df = pd.read_csv(path, sep=sep, header=None, on_bad_lines='skip')
            if df.shape[1] >= 2:
                df.columns = ['text', 'intent']
                return df
        raise ValueError(f"Không thể đọc đúng định dạng file: {path}")

    df_train = read_file(train_path)
    df_val = read_file(val_path)
    df_test = read_file(test_path)

    le = LabelEncoder()
    le.fit(pd.concat([df_train['intent'], df_val['intent'], df_test['intent']]))

    y_train = le.transform(df_train['intent'])
    y_val = le.transform(df_val['intent'])
    y_test = le.transform(df_test['intent'])
    num_classes = len(le.classes_)

    return df_train, df_val, df_test, y_train, y_val, y_test, num_classes, le


# =====================================================
# NHIỆM VỤ 1: TF-IDF + LOGISTIC REGRESSION
# =====================================================
def run_tfidf_lr(df_train, df_test, y_train, y_test, label_encoder):
    model = make_pipeline(
        TfidfVectorizer(max_features=7000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, C=3)
    )
    model.fit(df_train['text'], y_train)
    y_pred = model.predict(df_test['text'])
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    f1 = f1_score(y_test, y_pred, average='macro')
    return model, f1


# =====================================================
# NHIỆM VỤ 2: WORD2VEC TRUNG BÌNH + DENSE LAYER (CẢI THIỆN)
# =====================================================
def run_w2v_dense(df_train, df_val, df_test, y_train, y_val, y_test, num_classes):
    sentences = [text.split() for text in df_train['text']]
    w2v_model = Word2Vec(sentences, vector_size=150, window=5, min_count=2, workers=4, epochs=20)

    def sentence_to_avg_vector(text, model):
        words = text.split()
        vectors = [model.wv[w] for w in words if w in model.wv]
        return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

    X_train = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_train['text']])
    X_val = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_val['text']])
    X_test = np.array([sentence_to_avg_vector(t, w2v_model) for t in df_test['text']])

    model = Sequential([
        Input(shape=(w2v_model.vector_size,)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=30, batch_size=32, callbacks=[early_stop], verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro')
    return model, w2v_model, f1, loss


# =====================================================
# NHIỆM VỤ 3: LSTM VỚI EMBEDDING PRETRAINED (TỐI ƯU)
# =====================================================
def run_lstm_pretrained(df_train, df_val, df_test, y_train, y_val, y_test, w2v_model, num_classes):
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(df_train['text'])
    vocab_size = len(tokenizer.word_index) + 1
    max_len = 50

    def prepare_sequences(df):
        seq = tokenizer.texts_to_sequences(df['text'])
        return pad_sequences(seq, maxlen=max_len, padding='post')

    X_train = prepare_sequences(df_train)
    X_val = prepare_sequences(df_val)
    X_test = prepare_sequences(df_test)

    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=True),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=25, batch_size=32, callbacks=[early_stop], verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro')
    return model, tokenizer, max_len, f1, loss


# =====================================================
# NHIỆM VỤ 4: LSTM HỌC EMBEDDING TỪ ĐẦU (TỐI ƯU)
# =====================================================
def run_lstm_scratch(df_train, df_val, df_test, y_train, y_val, y_test, tokenizer, max_len, num_classes):
    def prepare_sequences(df):
        seq = tokenizer.texts_to_sequences(df['text'])
        return pad_sequences(seq, maxlen=max_len, padding='post')

    X_train = prepare_sequences(df_train)
    X_val = prepare_sequences(df_val)
    X_test = prepare_sequences(df_test)

    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, 128),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=25, batch_size=32, callbacks=[early_stop], verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro')
    return model, f1, loss



# =====================================================
# NHIỆM VỤ 5: ĐÁNH GIÁ VÀ PHÂN TÍCH
# =====================================================
def qualitative_analysis(models, tokenizer, le):
    examples = [
        "can you remind me to not call my mom",
        "is it going to be sunny or rainy tomorrow",
        "find a flight from new york to london but not through paris"
    ]
    print("\n=== PHÂN TÍCH ĐỊNH TÍNH ===")
    for ex in examples:
        print(f"\nCâu: {ex}")
        tfidf_pred = models["tfidf"].predict([ex])[0]
        print("TF-IDF + LR →", le.inverse_transform([tfidf_pred])[0])

        vec = np.mean([models["w2v"].wv[w] for w in ex.split() if w in models["w2v"].wv], axis=0)
        dense_pred = np.argmax(models["dense"].predict(vec.reshape(1, -1)), axis=1)[0]
        print("Word2Vec + Dense →", le.inverse_transform([dense_pred])[0])

        seq = tokenizer.texts_to_sequences([ex])
        pad = pad_sequences(seq, maxlen=50, padding='post')
        pre_pred = np.argmax(models["lstm_pre"].predict(pad), axis=1)[0]
        print("Pre-trained LSTM →", le.inverse_transform([pre_pred])[0])

        scratch_pred = np.argmax(models["lstm_scratch"].predict(pad), axis=1)[0]
        print("Scratch LSTM →", le.inverse_transform([scratch_pred])[0])


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    df_train, df_val, df_test, y_train, y_val, y_test, num_classes, le = load_and_prepare_data()

    tfidf_model, f1_lr = run_tfidf_lr(df_train, df_test, y_train, y_test, le)
    dense_model, w2v_model, f1_dense, loss_dense = run_w2v_dense(df_train, df_val, df_test, y_train, y_val, y_test, num_classes)
    lstm_pre, tokenizer, max_len, f1_pre, loss_pre = run_lstm_pretrained(df_train, df_val, df_test, y_train, y_val, y_test, w2v_model, num_classes)
    lstm_scratch, f1_scratch, loss_scratch = run_lstm_scratch(df_train, df_val, df_test, y_train, y_val, y_test, tokenizer, max_len, num_classes)

    results = pd.DataFrame([
        ["TF-IDF + Logistic Regression", f1_lr, "N/A"],
        ["Word2Vec (Avg) + Dense", f1_dense, loss_dense],
        ["Embedding (Pre-trained) + LSTM", f1_pre, loss_pre],
        ["Embedding (Scratch) + LSTM", f1_scratch, loss_scratch]
    ], columns=["Pipeline", "F1-score (Macro)", "Test Loss"])

    print("\n=== KẾT QUẢ ĐỊNH LƯỢNG (CẢI THIỆN) ===")
    print(results)

    models = {
        "tfidf": tfidf_model,
        "w2v": w2v_model,
        "dense": dense_model,
        "lstm_pre": lstm_pre,
        "lstm_scratch": lstm_scratch
    }

    qualitative_analysis(models, tokenizer, le)
