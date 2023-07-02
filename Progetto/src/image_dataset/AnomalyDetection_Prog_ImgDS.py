import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras import Model
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from load_img_dataset import load_img_dataset, load_normal_class_data
from skimage.transform import resize
from src.path_folders import DRIVE_FOLDER


def hist_plot(history_dict, title):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)  # 2 3
    plt.figure(num=0, figsize=(10, 10))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

# COME PRIMA COSA VADO A CARICARE I DATI

SEED = 4040

image_size = (60, 80)
batch_size = 32
split = .2

np.random.seed(SEED)
tf.random.set_seed(SEED)

x_train, x_test = load_normal_class_data(split)

x_full, y_full = load_img_dataset(anomaly_detection=True)

# DATA EXPLORATION: VADO A VISUALIZZARE IL NUMERO DI ELEMENTI PER CLASSE

normal_data = (y_full == 0).sum()
anomaly_data = (y_full == 1).sum()
dct = {'Normal Data (Shirts)': normal_data, 'Anomaly Data (Shorts, Sunglasses, Wallets)': anomaly_data}
df = pd.DataFrame.from_dict(dct, orient='index')


plt.figure(num=1, figsize=(15, 15))
totale = y_full.size
plt.title("Numerosità Classi Dataset Immagini (Anomaly Detection): Totale "+str(totale)+" Elementi")
graph = sns.barplot(x=df.index, y=df[0].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.3, df[0].values[i], ha="center")
    i += 1
plt.show()

# VADO AD EFFETTUARE LA RESIZE DELLA DIMENSIONE DELLE IMMAGINI DEL DATASET IN MODO DA POTER COSTRUIRE UNA RETE MAGGIORMENTE PROFONDA

size_dataset = x_full.shape[0]
new_img_size = (64, 80)

x_train_reshape, x_test_reshape = np.ones((x_train.shape[0], 64, 80, 3)), np.ones((x_test.shape[0], 64, 80, 3))

for i in range(x_train.shape[0]):
    frame = x_train[i]
    frame = resize(frame, new_img_size)
    x_train_reshape[i] = frame

for i in range(x_test.shape[0]):
    frame = x_test[i]
    frame = resize(frame, new_img_size)
    x_test_reshape[i] = frame

x_full_new = np.ones((size_dataset, 64, 80, 3))

for i in range(size_dataset):
    frame = x_full[i]
    frame = resize(frame, new_img_size)
    x_full_new[i] = frame

# LA SCELTA PER LA RISOLUZIONE DEL PROBELMA DI ANOMALY DETECTION E' QUELLA DI IMPIEGARE UNA RETE CNN IMPLEMENTANDO UN AUTOENCODER CONVOLUZIONALE, VADO A DEFINIRE IL MODELLO

# ENCODER:
inp = keras.Input(shape=(64, 80, 3))
x = Conv2D(filters=64, kernel_size=(9, 9), padding='same', activation='relu')(inp)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

# DECODER:
x = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(filters=32, kernel_size=(9, 9), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(filters=64, kernel_size=(9, 9), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D(size=(2, 2))(x)
decoded = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# TRAINING DELl' AUTOENCODER

EPOCHS = 50
VERBOSE = 2

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'autoencoder2.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder2.hist'), 'rb') as hist_file:
        hd = pickle.load(hist_file)
    autoencoder.load_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_w2'))
else:
    h = autoencoder.fit(x_train_reshape, x_train_reshape, epochs=EPOCHS, batch_size=batch_size, validation_data=(x_test_reshape, x_test_reshape), verbose=VERBOSE)
    hd = h.history
    with open(os.path.join(DRIVE_FOLDER, 'autoencoder2.hist'), 'wb') as hist_file:
        pickle.dump(hd, hist_file)
    autoencoder.save_weights(os.path.join(DRIVE_FOLDER, 'autoencoder_w2'))

# VADO A PLOTTARE I RISULTATI

hist_plot(hd, 'Loss Over Epochs')

# VADO A PLOTTARE LE IMMAGINI ORIGINALI E GENERATE

gen = np.zeros((25, 64, 80, 3))

plt.figure(num=2, figsize=(10, 10))
plt.suptitle('Original Images (Normal Class)')
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train_reshape[i])
    gen[i] = autoencoder.predict(x_train_reshape[i].reshape((1, 64, 80, 3)))

plt.figure(num=3, figsize=(10, 10))
plt.suptitle('Generated Images (Normal Class)')
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gen[i])

plt.show()

# EFFETTUO IL TESTING DEL MODELLO: ETICHETTA 0 = CLASSE NORMALE; 1 = CLASSE ANOMALY

outlierness = np.zeros(size_dataset)

# VADO A CALCOLARE L'OUTLIERNESS SCORE E PLOTTO LA CURVA ROC

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'outlierness'))
if exists:
    #**RISULTATI OTTENUTI: 0.9; IN CASO SI DOVESSERO AVERE RISULTATI DIFFERENTI ELIMINARE IL FILE 'outlierness' E RIEFFETTUARE IL CALCOLO DELLO SCORE
    with open(os.path.join(DRIVE_FOLDER, 'outlierness'), 'rb') as out_file:
        outlierness = pickle.load(out_file)
else:
    for i in range(size_dataset):
        outlierness[i] = autoencoder.evaluate(x_full_new[i].reshape((1, 64, 80, 3)), x_full_new[i].reshape((1, 64, 80, 3)), verbose=1)
    with open(os.path.join(DRIVE_FOLDER, 'outlierness'), 'wb') as out_file:
        pickle.dump(outlierness, out_file)

fpr, tpr, thresholds = roc_curve(y_full, outlierness)
auc = roc_auc_score(y_full, outlierness)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('AUC = '+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# VADO A MOSTRARE LE DIFFERENZE TRA LE IMMAGINI ORIGINALI E GENERATE PER QUANTO RIGUARDA LE ANOMALIE

gen_an = np.zeros((25, 64, 80, 3))

anomaly = x_full_new[y_full == 1]

plt.figure(num=5, figsize=(10, 10))
plt.suptitle('Original Images (Anomaly Class)')
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(anomaly[i])
    gen[i] = autoencoder.predict(anomaly[i].reshape((1, 64, 80, 3)))

plt.figure(num=6, figsize=(10, 10))
plt.suptitle('Generated Images (Anomaly Class)')
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gen[i])

plt.show()

# VADO A GRAFICARE LA MEDIA DI OUTLIERNESS SCORE PER CLASSE

diz = {'Normal Class (Shirts)': np.round(outlierness[y_full == 0].mean(), 7), 'Anomaly Class (Shorts, Sunglasses, Wallets)': np.round(outlierness[y_full == 1].mean(), 7)}
df = pd.DataFrame.from_dict(diz, orient='index')

plt.figure(figsize=(15, 15))
plt.title("Mean Loss per Class")
graph = sns.barplot(x=df.index, y=df[0].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.0001, df[0].values[i], ha="center")
    i += 1
plt.show()

# VADO A SELEZIONARE UN VALORE DI THRESHOLD

idx = np.argwhere(tpr >= .9)[0].item()

false_pos_rate, true_pos_rate, t = fpr[idx], tpr[idx], thresholds[idx]
print(f'Valore di Threshold selezionato t={t}; per tale soglia si ottiene TPR={true_pos_rate} e FPR={false_pos_rate}\n')

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot([false_pos_rate], [true_pos_rate], 'ro')
plt.annotate('t = '+str(t), xy=(false_pos_rate, true_pos_rate), xytext=(false_pos_rate-.2, true_pos_rate+.1), arrowprops=dict(facecolor='green', shrink=0.05))
plt.title('AUC = '+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(range((y_full == 0).sum()), outlierness[y_full == 0], 'b.', label='Normal Data')      # VADO AD EFFETTUARE I DUE PLOT IN MODO DA DISTINGUERE TRAMITE LEGENDA I DATI DELLA CLASSE NORMALE DALLE ANOMALIE
plt.plot(range((y_full == 0).sum(), y_full.size), outlierness[y_full == 1], 'r.', label='Anomaly Data')
plt.axhline(y=t, color='g', linestyle='-', label='Threshold')
plt.legend(loc='upper left')
plt.xlabel('Test ID')
plt.ylabel('Outlierness Score')

# VADO AD EFFETTUARE LA CLASSIFICAZIONE E CALCOLO IL CLASSIFICATION REPORT

y_pred = outlierness.copy()
y_pred = np.where(y_pred <= t, 0, 1)  # DOVE L'OUTLIERNESS SCORE E' INFERIORE O UGUALE ALLA SOGLIA LA CLASSE E' QUELLA NORMALE, ALTRIMENTI SI TRATTA DI ANOMALIE

report = classification_report(y_full, y_pred, target_names=['Normal', 'Anomaly'], output_dict=False)
report_dict = classification_report(y_full, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("Classification Report:")
print(report)

fig = plt.figure(figsize=(10, 10))
axis = fig.add_subplot(111)
w = 0.3
x_ticks = np.arange(2)
r1 = axis.bar(x_ticks-w, report_df['precision'].values[0:2], width=w, color='b', align='center', label='Precision')
r2 = axis.bar(x_ticks, report_df['recall'].values[0:2], width=w, color='g', align='center', label='Recall')
r3 = axis.bar(x_ticks+w, report_df['f1-score'].values[0:2], width=w, color='r', align='center', label='F1-Score')
axis.set_ylabel('Scores')
axis.set_title('Anomaly Detection: Classification Report \n Accuracy: '+str(np.round(report_df.loc['accuracy'][0], 2)))
axis.set_xticks(x_ticks)
axis.set_xticklabels(['Normal', 'Anomaly'])
axis.legend(loc='upper right')
axis.bar_label(r1, padding=3)
axis.bar_label(r2, padding=3)
axis.bar_label(r3, padding=3)