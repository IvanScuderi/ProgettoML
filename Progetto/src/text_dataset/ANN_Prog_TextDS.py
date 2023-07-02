import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Conv1D, Dense, GlobalMaxPool1D, GlobalAvgPool1D, Dropout, SpatialDropout1D, Embedding, LSTM, Bidirectional, BatchNormalization, SimpleRNN
from keras import Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score
from load_text_dataset import train_test_load_intindex
from src.path_folders import DRIVE_FOLDER

# FUNZIONI DI UTILITY

def hist_plot(history_dict, title):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    e = range(1, len(acc) + 1)

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(e, acc, 'b', label='Training acc')
    ax1.plot(e, val_acc, 'r-', label='Validation acc')
    ax1.set_title('Accuracy Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_xticks(e)
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(e, loss, 'b', label='Training loss')
    ax2.plot(e, val_loss, 'r-', label='Validation loss')
    ax2.set_title('Loss Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_xticks(e)
    ax2.set_ylabel('Loss')
    ax2.legend(loc='lower right')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def report_plot(r_df, title):
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    w = 0.3
    x_ticks = np.arange(len(classes))
    r1 = axis.bar(x_ticks-w, r_df['precision'].values[0:len(classes)], width=w, color='b', align='center', label='Precision')
    r2 = axis.bar(x_ticks, r_df['recall'].values[0:len(classes)], width=w, color='g', align='center', label='Recall')
    r3 = axis.bar(x_ticks+w, r_df['f1-score'].values[0:len(classes)], width=w, color='r', align='center', label='F1-Score')
    axis.set_ylabel('Scores')
    axis.set_title(title+' Classification Report \nTest Set Accuracy: '+str(np.round(r_df.loc['accuracy'][0], 4)))
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(classes)
    axis.legend(loc='upper right')
    axis.bar_label(r1, padding=3)
    axis.bar_label(r2, padding=3)
    axis.bar_label(r3, padding=3)
    fig.tight_layout()
    plt.show()

# VADO A CARICARE IL DATASET SU CUI HO PRECEDENTEMENTE EFFETTUATO PREPROCESSING
# PER FARE CIO' VADO A DEFINIRE ALCUNI PARAMETRI CHE UITLIZZERO' POI NELLA CREAZIONE DEL MIO MODELLO

SEED = 4040
np.random.seed(SEED)
tf.random.set_seed(SEED)

max_tokens = 20000 # NUMERO MASSIMO DI TOKEN CHE SI CONSIDERANO IN BASE ALLE LORO OCCORRENZE NEI RECORD
sequence_len = 50 # UNIFORMO LA DIMENSIONE DI OGNI SINGOLO RECORD, EVENTUALMENTE SI EFFETTUERA' PADDING CON 0 SE I RECORD NON CONTENGONO TALE NUMERO DI TOKEN
split = .15

# CODIFICA DATASET TRAIN/TEST: VETTORI NUMERICI DI INDICI DI PAROLE ALL'INTERNO DEL VOCABOLARIO ( LA LEN DI vocabulary E' PARI A max_tokens PER QUANTO DETTO)
# SCELTA FORZATA: LA VETTORIZZAZIONE TRAMITE TECNICA TFIDF ANDAVA A GENERARE UN DATASET CON UNA DIMENSIONALITA' TROPPO ELEVATA (OLTRE 30K COLONNE),
#  CHE MI HA PORTATO A PROBLEMI DI MEMORIA DURANTE LA COSTRUZIONE E TRAIN DEL MODELLO
x_train, x_test, y_train, y_test, classes, vocabulary = train_test_load_intindex(max_features=max_tokens, sequence_len=sequence_len, split=split, reduce_labels=True, report=True, return_other=True)

print(f'Esempio record del dataset: {x_train[0]}')

# VADO A COSTRUIRE UN PRIMO MODELLO COMPLESSO DI RETE: BIDIRECTIONAL + LSTM + CONV1D
def create_model_lstm():
    kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
    kernel_regularizer = keras.regularizers.l2()

    model = Sequential()
    model.add(Embedding(input_dim=max_tokens + 1, output_dim=128, input_length=sequence_len, trainable=True))
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
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# VADO ORA A PROVARE ALTRI MODELLI DI RETE:
# ---------------- RNN ----------------
def create_model_rnn():
    kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
    kernel_regularizer = keras.regularizers.l2()

    model = Sequential()
    model.add(Embedding(input_dim=max_tokens + 1, output_dim=128, input_length=sequence_len, trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(SimpleRNN(128, dropout=0.3, return_sequences=True))
    model.add(SimpleRNN(128, dropout=0.3, return_sequences=True))
    model.add(SpatialDropout1D(0.3))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalMaxPool1D())
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ---------------- CONV1D ----------------
def create_model_conv1d():
    kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
    kernel_regularizer = keras.regularizers.l2()

    model = Sequential()
    model.add(Embedding(input_dim=max_tokens + 1, output_dim=128, input_length=sequence_len, trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAvgPool1D())
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ---------------- DENSE ----------------
def create_model_dense():
    kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
    kernel_regularizer = keras.regularizers.l2()

    model = Sequential()
    model.add(Embedding(input_dim=max_tokens + 1, output_dim=128, input_length=sequence_len, trainable=True))
    model.add(keras.layers.Flatten())
    model.add(Dense(1250, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# VADO AD EFFETTUARE 10 FOLDS CV SUI MODELLI DEFINITI
                                            #pickle file name - epoche - batch_size
models = {"Bidirectional + LSTM + Conv1D": ["cv_bidir_lstm.hist", 5, 32],
          "RNN + Conv1D": ["cv_rnn.hist", 5, 128],
          "Conv1D": ["cv_text_conv1d.hist", 5, 32],
          "Dense": ["cv_text_dense.hist", 5, 32]
          }
val_accuracy_media = {"Bidirectional + LSTM + Conv1D": -1,
                      "RNN + Conv1D": -1,
                      "Conv1D": -1,
                      "Dense": -1
                      }

for k in models.keys():
    v = models[k]
    pickle_name = v[0]
    epochs = v[1]
    batch_size = v[2]

    # VADO AD APPLICARE 10 FOLDS CV

    exist = os.path.isfile(os.path.join(DRIVE_FOLDER, pickle_name))
    if exist:
        with open(os.path.join(DRIVE_FOLDER, pickle_name), 'rb') as hist_file:
            cv_results = pickle.load(hist_file)
    else:
        FOLDS = 10
        VERBOSE = 1

        classifier = None
        if k == "Bidirectional + LSTM + Conv1D":
            classifier = KerasClassifier(build_fn=lambda: create_model_lstm(), epochs=epochs, verbose=VERBOSE, batch_size=batch_size)
        if k == "RNN + Conv1D":
            classifier = KerasClassifier(build_fn=lambda: create_model_rnn(), epochs=epochs, verbose=VERBOSE, batch_size=batch_size)
        if k == "Conv1D":
            classifier = KerasClassifier(build_fn=lambda: create_model_conv1d(), epochs=epochs, verbose=VERBOSE, batch_size=batch_size)
        if k == "Dense":
            classifier = KerasClassifier(build_fn=lambda: create_model_dense(), epochs=epochs, verbose=VERBOSE, batch_size=batch_size)
        kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
        cv_results = cross_val_score(classifier, x_train, y_train, cv=kfold, verbose=VERBOSE)
        with open(os.path.join(DRIVE_FOLDER, pickle_name), 'wb') as hist_file:
            pickle.dump(cv_results, hist_file)

    # VADO A PLOTTARE I RISULTATI OTTENUTI
    val_accuracy = np.round(cv_results.mean(), 4)
    val_accuracy_media[k] = val_accuracy

    plt.figure(figsize=(10, 10))
    plt.ylim([0, 1])
    plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
    plt.title(f'{k} \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {val_accuracy}; Epoche = {epochs}; Batch Size = {batch_size}')
    plt.xlabel('Folds')
    plt.xticks(range(1, 11))
    plt.ylabel('Validation Accuracy')
    plt.show()

# VADO A PLOTTARE I VALORI DI VAL_ACCURACY MEDIA

df = pd.DataFrame.from_dict(val_accuracy_media, orient='index')

plt.figure(figsize=(15, 15))
plt.title("Validation Accuracy media sui 10 fold per modello")
graph = sns.barplot(x=df.index, y=df[0].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.001, df[0].values[i], ha="center")
    i += 1
plt.show()

# VADO ORA AD ANALIZZARE IL MODELLO MIGLIORE: BIDIRECTIONAL + LSTM + CONV1D
# PERCHE' SI SVILUPPANO SOLO 3 EPOCHE PER LA FASE DI TRAINING? VADO A PROVARE IL TRAINING CON UN NUMERO MAGGIORE DI EPOCHE

m = create_model_lstm()

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'test_lstm.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'test_lstm.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
else:
    EPOCHS = 10
    VERBOSE = 1

    h = m.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_split=.2, verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'test_lstm.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)

hist_plot(hd, 'LSTM (10 Epochs)')

#  SVILUPPO LE 5 EPOCHE DI TRAINING E SALVO IL MODELLO OTTENUTO

m = create_model_lstm()

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'model_lstm_text.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'model_lstm_text.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    m.load_weights(os.path.join(DRIVE_FOLDER, 'model_w_lstm'))
else:
    EPOCHS = 5
    VERBOSE = 1

    h = m.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_split=.2, verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'model_lstm_text.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    m.save_weights(os.path.join(DRIVE_FOLDER, 'model_w_lstm'))

hist_plot(hd, 'LSTM (5 Epochs)')

# EFFETTUO IL TESTING DEL MODELLO CALCOLANDO IL CLASSIFICATION REPORT SUL DATASET DI TEST

predictions = m.predict(x_test)
y_pred = predictions.argmax(axis=1)

report = classification_report(y_test, y_pred, target_names=classes, output_dict=False)
report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("Classification Report LSTM:")
print(report)

# VADO A PLOTTARE I RISULTATI

report_plot(report_df, 'LSTM')