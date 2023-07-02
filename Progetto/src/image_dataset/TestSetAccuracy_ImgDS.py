import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from skimage.color import rgb2gray
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Rescaling
from keras import Model
from load_img_dataset import load_img_dataset
from src.path_folders import DS_FOLDER_IMG, DRIVE_FOLDER

# SCRIPT CHE CALCOLA L'ACCURATEZZA DEI MODELLI SUL TEST SET PER QUANTO RIGUARDA IL DATASET DI IMMAGINI

SEED = 4040

image_size = (60, 80)
batch_size = 32
split = .2
class_names = ['Shirts', 'Shorts', 'Sunglasses', 'Wallets']

np.random.seed(SEED)
tf.random.set_seed(SEED)

# VADO AD IMPORTARE I DATASET PER COME SONO STATI UTILIZZATI DURANTE LA FASE DI TRAINING DEI VARI MODELLI
# LA RETE CNN IMPIEGA IL DATASET CON IMMAGINI A 3 CANALI COLORE MENTRE PER GLI ESTIMATOR DI SKLEARN SI IMPIEGA CONVERSIONE IN SCALA DI GRIGI

_, x_test_rgb, _, y_test = load_img_dataset(split=split, report=True)

val_ds = keras.preprocessing.image_dataset_from_directory(DS_FOLDER_IMG,
                                                          seed=SEED,
                                                          image_size=image_size,
                                                          batch_size=batch_size,
                                                          class_names=class_names,
                                                          validation_split=split,
                                                          subset='validation'
                                                          )

tmp_test = []
for img in x_test_rgb:
    gray_img = rgb2gray(img)
    tmp_test.append(gray_img)

tmp_test = np.asarray(tmp_test)
x_test_gs = tmp_test.reshape(tmp_test.shape[0], 60*80)

# VADO ORA A CARICARE I MODELLI
# 1): CNN

def create_model(cv=True):
    inp = keras.Input(shape=(60, 80, 3))
    if not cv:
        x = Rescaling(1. / 255)(inp)
    else:
        # non mi serve effettuare Rescaling perchè l'input è già nell'intervallo [0, 1] quando impiego il training set importato tramite metodo custom load_img_dataset
        x = inp
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = MaxPooling2D(strides=2)(x)
    x = Conv2D(filters=48, kernel_size=(5, 5), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = MaxPooling2D(strides=2)(x)
    x = Conv2D(filters=48, kernel_size=(5, 5), padding='valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = MaxPooling2D(strides=2)(x)
    x = Flatten()(x)
    x = Dense(1344, activation='relu', kernel_regularizer=keras.regularizers.l2())(x)
    x = Dropout(0.3)(x)
    x = Dense(672, activation='relu', kernel_regularizer=keras.regularizers.l2())(x)
    x = Dropout(0.3)(x)
    x = Dense(560, activation='relu', kernel_regularizer=keras.regularizers.l2())(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2())(x)
    x = Dropout(0.3)(x)
    y = Dense(4, activation='softmax')(x)

    m = Model(inp, y)

    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return m

cnn = create_model(cv=False)
cnn.load_weights(os.path.join(DRIVE_FOLDER, 'model_w2'))

# 2): ADABOOST

with open(os.path.join(DRIVE_FOLDER, 'ada_grid_search.hist'), 'rb') as hist_file:
    ada_grid_search_CV = pickle.load(hist_file)

ada_boost = ada_grid_search_CV.best_estimator_

# 3): SVM

with open(os.path.join(DRIVE_FOLDER, 'svc_grid_search.hist'), 'rb') as hist_file:
    svc_grid_search_CV = pickle.load(hist_file)

svc = svc_grid_search_CV.best_estimator_

# 4): DENSITY ESTIMATION - KNN

with open(os.path.join(DRIVE_FOLDER, 'knn_grid_search.hist'), 'rb') as hist_file:
    knn_grid_search_CV = pickle.load(hist_file)

de_knn = knn_grid_search_CV.best_estimator_

# CALCOLO I VALORI DI ACCURATEZZA SUL TEST SET

models_name = ['CNN', 'AdaBoost', 'SVC', 'Density Estimation (KNN)']
accuracy = []

#1): CNN
predictions = []
y_true = []

for img, labels in val_ds:
    if img.shape[0] != batch_size:
        # EVITO PROBLEMI LEGATI ALL'ULTIMO BATCH CHE POTREBBE AVERE DIMENSIONE INFERIORE A batch_size
        break
    y_pred = cnn(img)
    _, idx = tf.math.top_k(y_pred, k=1)
    idx = idx.numpy()
    idx = idx.reshape(img.shape[0])
    y_true.append(labels.numpy())
    predictions.append(idx)

predictions = np.asarray(predictions)
predictions = predictions.ravel()

y_true = np.asarray(y_true)
y_true = y_true.ravel()

accuracy.append(np.round(accuracy_score(predictions, y_true), 5))

#OTHERS):
models = [ada_boost, svc, de_knn]

for model in models:
    y_pred = model.predict(x_test_gs)
    accuracy.append(np.round(accuracy_score(y_test, y_pred), 5))


# VADO A PLOTTARE I RISULTATI OTTENUTI

zip_iterator = zip(models_name, accuracy)
ris_dict = dict(zip_iterator)

df = pd.DataFrame.from_dict(ris_dict, orient='index')

plt.figure(figsize=(15, 15))
plt.title("Image Dataset: Test Set Accuracy")
graph = sns.barplot(x=df.index, y=df[0].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.01, df[0].values[i], ha="center")
    i += 1
plt.show()
