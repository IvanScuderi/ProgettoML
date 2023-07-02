import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import resize
import imageio
from src.path_folders import DS_FOLDER_IMG

SEED = 4040

class_label = {'Shirts': 0, 'Shorts': 1, 'Sunglasses': 2, 'Wallets': 3}
class_names = ['Shirts', 'Shorts', 'Sunglasses', 'Wallets']


def load_img_dataset(split=0.2, report=False, anomaly_detection=False):
    # SCRIPT CHE MI PERMETTE DI CARICARE IL DATASET DI IMMAGINI COME NDARRAY DAL DISCO

    image_size = (60, 80)

    totale = 0
    if report:
        elem_per_class = {}
        for folder in os.listdir(DS_FOLDER_IMG):
            len_folder = len(os.listdir(os.path.join(DS_FOLDER_IMG, folder)))
            elem_per_class[folder] = len_folder
            totale += len_folder

        df = pd.DataFrame.from_dict(elem_per_class, orient='index')

        plt.figure(figsize=(15, 15))
        plt.title("Numerosità Classi Dataset Immagini: Totale "+str(totale)+" Elementi")
        graph = sns.barplot(x=df.index, y=df[0].values)
        i = 0
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x()+p.get_width()/2., height + 0.1, df[0].values[i], ha="center")
            i += 1
        plt.show()

    # VADO A LEGGERE IL DATASET
    x = []
    y = []
    error = []
    error_label = {'Shirts': 0,
                   'Shorts': 0,
                   'Sunglasses': 0,
                   'Wallets': 0
                   }

    for folder in os.listdir(DS_FOLDER_IMG):
        path_to_folder = os.path.join(DS_FOLDER_IMG, folder)
        for elem in os.listdir(path_to_folder):
            frame = imageio.imread(os.path.join(path_to_folder, elem))
            if len(frame.shape) != 3:
                error.append(elem)
                error_label[folder] += 1
                continue
            frame = resize(frame, image_size)
            x.append(frame)
            if not anomaly_detection:
                y.append(class_label[folder])
            else:
                if folder == "Shirts":
                    # CLASSE INLIER -> SHIRTS
                    y.append(0)
                else:
                    # CLASSE OUTLIER -> OTHERS
                    y.append(1)

    x = np.asarray(x)
    y = np.asarray(y)

    if report:
        print(f'Dimensione totale dataset caricato: X -> {x.shape} , Y -> {y.shape}')
        print(f'Numero di immagini che non è stato possibile caricare per problemi di lettura: {totale-len(x)}')
        print(f'Immagini su cui sono stati riscontrati problemi di lettura: {error}')
        print(f'Numero di immagini escluse per classe: {error_label} \n')

    if not anomaly_detection:
        # GENERO IL DATASET DI TRAIN E TEST
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=SEED)

        if report:
            print(f'Valore di split impiegato per la divisione in train-test: {split}')
            print(f'Dimensione totale Training Set: {x_train.shape}')
            print(f'Dimensione totale Test Set: {x_test.shape} \n')

        return x_train, x_test, y_train, y_test

    else:
        # NELL'AMBITO DELL'ANOMALY DETECTION VADO AD IMPIEGARE QUESTO METODO PER OTTENERE IL DATASET DI TEST
        # OSSIA IL DATASET CHE CONTIENE SIA ELEMENTI DELLA CLASSE INLIER CHE OUTLIER
        return x, y


def load_normal_class_data(split=.2):
    # SCRIPT CHE PERMETTE IL CARICAMENTO DEI DATI NORMALI, OSSIA LE IMG DI CLASSE SHIRTS CHE RISULTA ESSERE LA PIU' NUMEROSA,
    # PER POTER ANDARE A RISOLVERE IL PROBLEMA DI SEMI SUP ANOMALY DETECTION

    image_size = (60, 80)
    normal_class = 'Shirts'
    normal_path = os.path.join(DS_FOLDER_IMG, normal_class)

    x = []
    for elem in os.listdir(normal_path):
        frame = imageio.imread(os.path.join(normal_path, elem))
        frame = resize(frame, image_size)
        x.append(frame)

    x = np.asarray(x)

    x_train, x_test = train_test_split(x, test_size=split, random_state=SEED)

    print(f'Train/Test Set size: {x_train.shape[0]}/{x_test.shape[0]}; (split = {split}) \n')

    return x_train, x_test