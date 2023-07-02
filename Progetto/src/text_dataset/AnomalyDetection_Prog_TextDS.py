import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from keras.layers import Dense, Dropout, Embedding, BatchNormalization, GlobalMaxPool1D, SpatialDropout1D, Conv1D, LSTM, Bidirectional
from keras import Model, Input, Sequential
from load_text_dataset import load_dataset_anomaly_detection
from src.path_folders import DRIVE_FOLDER

def hist_plot(history_dict, title):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)  # 2 3
    plt.figure(num=0, figsize=(10, 10))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

# CARICAMENTO DEI DATI
SEED = 4040
np.random.seed(SEED)
tf.random.set_seed(SEED)

max_tokens = 1000
split = .15

# 'x_train' ED 'x_test' CONTENGONO UNICAMENTE ELEMENTI DELLA CLASSE NORMALE (POSITIVE), MENTRE X E Y CONTENGONO L'INTERO DATASET
x_train, x_test, X, Y, other = load_dataset_anomaly_detection('tfidf', max_features=max_tokens, split=split, return_other=True)

# VADO A DEFINIRE UN MODELLO DI RETE DENSO PER LA RISOLUZIONE DEL TASK DI AD

kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
kernel_regularizer = keras.regularizers.l2()
hidden_space = 128

# ENCODER:
inp = Input(shape=(max_tokens,))
x = Dense(1000, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inp)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
x = Dropout(0.3)(x)
x = Dense(hidden_space, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
encoded = Dropout(0.3)(x)
# DECODER:
x = Dense(hidden_space, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(encoded)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
x = Dropout(0.3)(x)
decoded = Dense(max_tokens, activation='sigmoid')(x)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# TRAINING DELl' AUTOENCODER

EPOCHS = 30
VERBOSE = 1
batch_size = 64

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'autoencoder_text.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder_text.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    autoencoder.load_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_w_text'))
else:
    h = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=batch_size, validation_data=(x_test, x_test), verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder_text.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    autoencoder.save_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_w_text'))

# VADO A PLOTTARE I RISULTATI

hist_plot(hd, f'Autoencoder Dense\nLoss Over Epochs:')

# EFFETTUO IL TESTING DEL MODELLO: ETICHETTA 0 = CLASSE NORMALE; 1 = CLASSE ANOMALY

outlierness = np.zeros(len(X))

# VADO A CALCOLARE L'OUTLIERNESS SCORE E PLOTTO LA CURVA ROC

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'outlierness_text'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'outlierness_text'), 'rb') as out_file:
        outlierness = pickle.load(out_file)
else:
    for i in range(len(X)):
        outlierness[i] = autoencoder.evaluate(X[i].reshape((1, -1)), X[i].reshape((1, -1)), verbose=1)
    with open(os.path.join(DRIVE_FOLDER, 'outlierness_text'), 'wb') as out_file:
        pickle.dump(outlierness, out_file)

fpr, tpr, thresholds = roc_curve(Y, outlierness)
auc = roc_auc_score(Y, outlierness)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title(f'Autoencoder Dense (N Tokens = {max_tokens}, Hidden Space = {hidden_space}, Epochs = {EPOCHS}):\nAUC = '+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# VADO A GRAFICARE LA MEDIA DI OUTLIERNESS SCORE PER CLASSE

diz = {'Normal Class (Positive)': np.round(outlierness[Y == 0].mean(), 7), 'Anomaly Class (Negative, Neutral)': np.round(outlierness[Y == 1].mean(), 7)}
df = pd.DataFrame.from_dict(diz, orient='index')

plt.figure(figsize=(15, 15))
plt.title("Mean Loss per Class")
graph = sns.barplot(x=df.index, y=df[0].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.000001, df[0].values[i], ha="center")
    i += 1
plt.show()

# APPENDICE 1):
# VADO A PROVARE UN ULTERIORE MODELLO BASATO SU RETE LSTM + VETTORIZZAZIONE LISTA DI INDICI

max_tokens = 30000
sequence_len = 50

x_train_int, x_test_int, X_int, Y_int, other_int = load_dataset_anomaly_detection('int', max_features=max_tokens, sequence_len=sequence_len, split=split, return_other=True, plot=False)

def create_model_anomaly_detection_lstm():
    model = Sequential()
    model.add(Embedding(input_dim=max_tokens + 1, output_dim=256, input_length=sequence_len, trainable=True))
    # ENCODER
    model.add(Bidirectional(LSTM(256, dropout=0.25, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, dropout=0.25, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(0.3))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(0.3))
    # DECODER
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalMaxPool1D())
    model.add(Dropout(.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(sequence_len, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss="mse",
    )

    return model

autoencoder_lstm = create_model_anomaly_detection_lstm()

EPOCHS = 20
VERBOSE = 1
batch_size = 32

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'autoencoder_lstm_text.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder_lstm_text.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    autoencoder_lstm.load_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_lstm_text_w'))
else:
    h = autoencoder_lstm.fit(x_train_int, x_train_int, epochs=EPOCHS, batch_size=batch_size, validation_data=(x_test_int, x_test_int), verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder_lstm_text.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    autoencoder_lstm.save_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_lstm_text_w'))

outlierness_int = np.zeros(len(X_int))

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'outlierness_lstm_text'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'outlierness_lstm_text'), 'rb') as out_file:
        outlierness_int = pickle.load(out_file)
else:
    for i in range(len(X_int)):
        outlierness_int[i] = autoencoder_lstm.evaluate(X_int[i].reshape((1, -1)), X_int[i].reshape((1, -1)), verbose=1)
    with open(os.path.join(DRIVE_FOLDER, 'outlierness_lstm_text'), 'wb') as out_file:
        pickle.dump(outlierness_int, out_file)

fpr_int, tpr_int, thresholds_int = roc_curve(Y_int, outlierness_int)
auc_int = roc_auc_score(Y_int, outlierness_int)

plt.figure(figsize=(10, 10))
plt.plot(fpr_int, tpr_int)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Autoencoder LSTM:\nAUC = '+str(auc_int))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# APPENDICE 2):
# VADO A VEDERE COME VARIA LA CURVA ROC E AUC SE MODIFICO IL NUMERO DI TOKEN DEL VOCABOLARIO DA CONSIDERARE E LO SPAZIO LATENTE NELL'ARCHITETTURA DENSA

max_tokens = 3000

x_train2, x_test2, X2, Y2, other2 = load_dataset_anomaly_detection('tfidf', max_features=max_tokens, split=split, return_other=True, plot=False)

def create_model_anomaly_detection_dense_test(hidden):
    # ENCODER:
    inp_t = Input(shape=(max_tokens,))
    x_t = Dense(max_tokens, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inp_t)
    x_t = Dropout(0.3)(x_t)
    x_t = Dense(512, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_t)
    x_t = Dropout(0.3)(x_t)
    x_t = Dense(hidden, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_t)
    encoded_t = Dropout(0.3)(x_t)
    # DECODER:
    x_t = Dense(hidden, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(encoded_t)
    x_t = Dropout(0.3)(x_t)
    x_t = Dense(512, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x_t)
    x_t = Dropout(0.3)(x_t)
    decoded_t = Dense(max_tokens, activation='sigmoid')(x_t)

    autoencoder_t = Model(inp_t, decoded_t)
    autoencoder_t.compile(optimizer='adam', loss='mse')

    return autoencoder_t

hidden_space = 256
autoencoder_2 = create_model_anomaly_detection_dense_test(hidden_space)

EPOCHS = 20
VERBOSE = 1
batch_size = 32

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'autoencoder_text_2.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder_text_2.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    autoencoder_2.load_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_text_2_w'))
else:
    h = autoencoder_2.fit(x_train2, x_train2, epochs=EPOCHS, batch_size=batch_size, validation_data=(x_test2, x_test2), verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder_text_2.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    autoencoder_2.save_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_text_2_w'))

outlierness_2 = np.zeros(len(X2))

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'outlierness_2_text'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'outlierness_2_text'), 'rb') as out_file:
        outlierness_2 = pickle.load(out_file)
else:
    for i in range(len(X2)):
        outlierness_2[i] = autoencoder_2.evaluate(X2[i].reshape((1, -1)), X2[i].reshape((1, -1)), verbose=1)
    with open(os.path.join(DRIVE_FOLDER, 'outlierness_2_text'), 'wb') as out_file:
        pickle.dump(outlierness_2, out_file)

fpr_2, tpr_2, thresholds_2 = roc_curve(Y2, outlierness_2)
auc_2 = roc_auc_score(Y2, outlierness_2)

plt.figure(figsize=(10, 10))
plt.plot(fpr_2, tpr_2)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title(f'Autoencoder Dense (N Tokens = {max_tokens}, Hidden Space = {hidden_space}, Epochs = {EPOCHS}):\nAUC = '+str(auc_2))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()