import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Rescaling
from keras import Model, Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score
from load_img_dataset import load_img_dataset
from src.path_folders import DS_FOLDER_IMG, DRIVE_FOLDER

# VADO A GENERARE UNA RETE CNN CHE ADDEDSTRO MEDIANTE L'UTILIZZO DI UN TENSOR DATASET

def hist_plot(history_dict, title):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)  # 2 3
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def report_plot(r_df, title):
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    w = 0.3
    x_ticks = np.arange(4)
    r1 = axis.bar(x_ticks-w, r_df['precision'].values[0:4], width=w, color='b', align='center', label='Precision')
    r2 = axis.bar(x_ticks, r_df['recall'].values[0:4], width=w, color='g', align='center', label='Recall')
    r3 = axis.bar(x_ticks+w, r_df['f1-score'].values[0:4], width=w, color='r', align='center', label='F1-Score')
    axis.set_ylabel('Scores')
    axis.set_title(title+' Classification Report \nTest Set Accuracy: '+str(np.round(r_df.loc['accuracy'][0], 4)))
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(class_names)
    axis.legend(loc='upper right')
    axis.bar_label(r1, padding=3)
    axis.bar_label(r2, padding=3)
    axis.bar_label(r3, padding=3)
    fig.tight_layout()
    plt.show()

SEED = 4040

image_size = (60, 80)
batch_size = 32
split = .2
class_names = ['Shirts', 'Shorts', 'Sunglasses', 'Wallets']

# TF.DATA.DATASET: OGGETTO DATASET CHE IMPIEGO PER IL TRAIN DELLA RETE CNN

train_ds = keras.preprocessing.image_dataset_from_directory(DS_FOLDER_IMG,
                                                            seed=SEED,
                                                            image_size=image_size,
                                                            batch_size=batch_size,
                                                            class_names=class_names,
                                                            validation_split=split,
                                                            subset='training'
                                                            )

val_ds = keras.preprocessing.image_dataset_from_directory(DS_FOLDER_IMG,
                                                          seed=SEED,
                                                          image_size=image_size,
                                                          batch_size=batch_size,
                                                          class_names=class_names,
                                                          validation_split=split,
                                                          subset='validation'
                                                          )

# DATASET DI IMMAGINI CHE IMPIEGO PER POTER UTILIZZARE LA CLASSE WRAPPER KERASCLASSIFIER E SFRUTTARE I METODI DELLA LIBRERIA SKLEARN PER LA CV

x_train, x_test, y_train, y_test = load_img_dataset(split=split, report=True)

# VADO A PLOTTARE ALCUNE IMMAGINI PER CONTROLLARE IL COMPLETAMENTO CORRETTO DELL'OPERAZIONE

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# VADO A DEFINIRE IL MIO MODELLO DI RETE CNN

# PER IMPIEGARE I WRAPPER DI KERAS PER SKLEARN NECESSITO DI DEFINIRE UN METODO CHE MI RESTITUISCA IL MODELLO DI RETE DA ADDESTRARE


def create_model_cnn(cv=True):
    #data_augmentation = keras.Sequential(
    #    [
    #        keras.layers.RandomFlip("horizontal"),
    #        keras.layers.RandomFlip("vertical"),
    #        keras.layers.RandomRotation(0.1),
    #    ]
    #)
    inp = keras.Input(shape=(60, 80, 3))
    # x = data_augmentation(inp) HO NOTATO PRESTAZIONI PEGGIORI USANDO DA
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

# VADO AD APPLICARE 10 FOLDS CV

np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 40
VERBOSE = 1
FOLDS = 10

# WRAPPER PER IMPIEGARE LA LIBRERIA SKLEARN CON MODELLI DELLA LIBRERIA TF E KERAS

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'cv2.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'cv2.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    classifier = KerasClassifier(build_fn=lambda: create_model_cnn(), epochs=EPOCHS, verbose=VERBOSE, batch_size=batch_size)
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(classifier, x_train, y_train, cv=kfold, verbose=VERBOSE)
    with open(os.path.join(DRIVE_FOLDER, 'cv2.hist'), 'wb') as hist_file:
        pickle.dump(cv_results, hist_file)


# VADO A PLOTTARE I RISULTATI OTTENUTI

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
plt.title(f'CNN \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# VADO A PROVARE UN MODELLO DI RETE DEEP SENZA USARE CONVOLUZIONE

def create_model_deep(cv=True):

    model = Sequential()

    kernel_initializer = keras.initializers.glorot_uniform(seed=np.random.randint(20))
    kernel_regularizer = keras.regularizers.l2()

    if not cv:
        model.add(Rescaling(1. / 255))

    model.add(Flatten(input_shape=(60, 80, 3)))
    model.add(Dense(2800, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(1200, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(256, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'cv_deep.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'cv_deep.hist'), 'rb') as hist_file:
        cv_results_deep = pickle.load(hist_file)
else:
    classifier_deep = KerasClassifier(build_fn=lambda: create_model_deep(), epochs=EPOCHS, verbose=VERBOSE, batch_size=batch_size)
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results_deep = cross_val_score(classifier_deep, x_train, y_train, cv=kfold, verbose=VERBOSE)
    with open(os.path.join(DRIVE_FOLDER, 'cv_deep.hist'), 'wb') as hist_file:
        pickle.dump(cv_results_deep, hist_file)

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results_deep, 'ro', range(1, 11), cv_results_deep, 'k--')
plt.title(f'Deep ANN \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results_deep.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# EFFETTUO IL TRAIN DEL MIO MODELLO CONVOLUZIONALE

model_cnn = create_model_cnn(cv=False)

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'model2.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'model2.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    model_cnn.load_weights(os.path.join(DRIVE_FOLDER, 'model_w2'))
else:
    h = model_cnn.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'model2.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    model_cnn.save_weights(os.path.join(DRIVE_FOLDER, 'model_w2'))

hist_plot(hd, 'CNN: Accuracy Over Epochs')

# EFFETTUO IL TRAIN DEL MIO MODELLO DEEP

model_deep = create_model_deep(cv=False)

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'model_deep.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'model_deep.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    model_deep.load_weights(os.path.join(DRIVE_FOLDER, 'model_w_deep'))
else:
    h = model_deep.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'model_deep.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    model_deep.save_weights(os.path.join(DRIVE_FOLDER, 'model_w_deep'))

hist_plot(hd, 'Deep ANN: Accuracy Over Epochs')

# VADO AD EFFETTUARE TESTING ANDANDO A CALCOLARE E PLOTTARE UN CLASSIFICATION REPORT

predictions_cnn = []
predictions_deep = []
y_true = []

for img, labels in val_ds:
    if img.shape[0] != batch_size:
        # EVITO PROBLEMI LEGATI ALL'ULTIMO BATCH CHE POTREBBE AVERE DIMENSIONE INFERIORE A batch_size
        break
    y_true.append(labels.numpy())

    y_pred_cnn = model_cnn(img)
    y_pred_deep = model_deep(img)

    _, idx = tf.math.top_k(y_pred_cnn, k=1)
    idx = idx.numpy()
    idx = idx.reshape(img.shape[0])  # BATCH SIZE
    predictions_cnn.append(idx)

    _, idx = tf.math.top_k(y_pred_deep, k=1)
    idx = idx.numpy()
    idx = idx.reshape(img.shape[0])  # BATCH SIZE
    predictions_deep.append(idx)

predictions_cnn = np.asarray(predictions_cnn)
predictions_cnn = predictions_cnn.ravel()

predictions_deep = np.asarray(predictions_deep)
predictions_deep = predictions_deep.ravel()

y_true = np.asarray(y_true)
y_true = y_true.ravel()

# CLASSIFICATION REPORT: MODEL CNN

report = classification_report(y_true, predictions_cnn, target_names=class_names, output_dict=False)
report_dict = classification_report(y_true, predictions_cnn, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# CLASSIFICATION REPORT: MODEL DEEP

report_deep = classification_report(y_true, predictions_deep, target_names=class_names, output_dict=False)
report_deep_dict = classification_report(y_true, predictions_deep, target_names=class_names, output_dict=True)
report_deep_df = pd.DataFrame(report_deep_dict).transpose()

print("Classification Report CNN:")
print(report)

print("Classification Report Deep ANN:")
print(report_deep)

# VADO A PLOTTARE I RISULTATI

report_plot(report_df, 'CNN')
report_plot(report_deep_df, 'Deep ANN')