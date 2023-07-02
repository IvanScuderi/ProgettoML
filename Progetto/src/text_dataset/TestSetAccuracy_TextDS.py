import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Conv1D, Dense, GlobalMaxPool1D, Dropout, SpatialDropout1D, Embedding, LSTM, Bidirectional, BatchNormalization
from keras import Sequential
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from load_text_dataset import train_test_load_tfidf, train_test_load_intindex
from src.path_folders import DRIVE_FOLDER

# SCRIPT CHE CALCOLA L'ACCURATEZZA DEI MODELLI SUL TEST SET PER QUANTO RIGUARDA IL DATASET DI TESTI

SEED = 4040
split = .15

np.random.seed(SEED)
tf.random.set_seed(SEED)

# CARICO IL DATASET DI TEST CHE IMPIEGO PER LA MIA RETE NEURALE
max_tokens_ann = 20000
sequence_len = 50

_, x_test_ann, _, y_test_ann = train_test_load_intindex(max_features=max_tokens_ann, sequence_len=sequence_len, split=split, reduce_labels=True, report=True, return_other=False)

# CARICO IL DATASET DI TEST CHE IMPIEGO PER I RESTANTI MODELLI
max_tokens = 500

_, x_test, _, y_test = train_test_load_tfidf(max_features=max_tokens,  split=split, reduce_labels=True, report=False, return_other=False)

# VADO ORA A CARICARE I MODELLI
# 1): Bidirectional + LSTM + Conv1D
def create_model_lstm():
    kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
    kernel_regularizer = keras.regularizers.l2()

    model = Sequential()
    model.add(Embedding(input_dim=max_tokens_ann + 1, output_dim=128, input_length=sequence_len, trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=True)))
    model.add(SpatialDropout1D(0.3))
    model.add(Conv1D(128, 7, padding='same', activation='relu'))
    model.add(Conv1D(128, 7, padding='same', activation='relu'))
    model.add(Conv1D(128, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalMaxPool1D())
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

ann = create_model_lstm()
ann.load_weights(os.path.join(DRIVE_FOLDER, 'model_w_lstm'))

# 2): ADABOOST

with open(os.path.join(DRIVE_FOLDER, 'ada_grid_search_text.hist'), 'rb') as hist_file:
    ada_grid_search_CV = pickle.load(hist_file)

ada_boost = ada_grid_search_CV.best_estimator_

# 3): SVM

with open(os.path.join(DRIVE_FOLDER, 'svc_grid_search_text.hist'), 'rb') as hist_file:
    svc_grid_search_CV = pickle.load(hist_file)

svc = svc_grid_search_CV.best_estimator_

# 4): DENSITY ESTIMATION - BERNOULLI NAIVE BAYES

with open(os.path.join(DRIVE_FOLDER, 'bnaiveb_grid_search_text.hist'), 'rb') as hist_file:
    bnaiveb_grid_search_CV = pickle.load(hist_file)

de_bnaiveb = bnaiveb_grid_search_CV.best_estimator_

# CALCOLO I VALORI DI ACCURATEZZA SUL TEST SET

models_name = ['ANN (Bidirectional + LSTM + Conv1D)', 'AdaBoost', 'SVC', 'Density Estimation (Bernoulli Naive Bayes)']
accuracy = []

#ANN):
predictions = ann.predict(x_test_ann)
y_pred = predictions.argmax(axis=1)
accuracy.append(np.round(accuracy_score(y_test_ann, y_pred), 2))

#OTHERS):
models = [ada_boost, svc, de_bnaiveb]

for m in models:
    y_pred = m.predict(x_test)
    accuracy.append(np.round(accuracy_score(y_test, y_pred), 2))

# VADO A PLOTTARE I RISULTATI OTTENUTI

zip_iterator = zip(models_name, accuracy)
ris_dict = dict(zip_iterator)

df = pd.DataFrame.from_dict(ris_dict, orient='index')

plt.figure(figsize=(15, 15))
plt.title("Text Dataset: Test Set Accuracy")
graph = sns.barplot(x=df.index, y=df[0].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.01, df[0].values[i], ha="center")
    i += 1
plt.show()

