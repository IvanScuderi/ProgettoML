import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization
import matplotlib.pyplot as plt
import seaborn as sns
from src.path_folders import DRIVE_FOLDER

# FUNZIONI CHE COMPLETANO LE OPERAZIONI DI PREPROCESSING ANDANDO A COSTRUIRE UN DATASET DI TRAIN/TEST CONTENENTE VALORI NUMERICI

SEED = 4040
sns.set()

def train_test_load_tfidf(max_features=None, split=.25, reduce_labels=False, report=False, return_other=False, scale=False):

    exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'))
    if exist:
        with open(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'), 'rb') as hf:
            df_load = pickle.load(hf)
    else:
        raise RuntimeError('Missing text_dataframe_preproc file, please run first "data_preprocessing.py" script to obtain this file.')

    # VADO A RAFFINARE IL PREPROCESSING DEI DATI PER I MODELLI NON BASATI SU RETI NEURALI
    if reduce_labels:
        df_load['Sentiment'] = df_load['Sentiment'].replace(['Extremely Negative', 'Extremely Positive'], ['Negative', 'Positive'])
        df_load = df_load.drop_duplicates(subset=['StemmedString'])

        # RIMUOVO I RECORD CHE CONTENGONO MENO DI 2 TOKEN
        token_len = df_load['StemmedTweet'].apply(len)
        df_load[token_len <= 2] = np.nan
        df_load = df_load.dropna()

        if report:
            gb = df_load.groupby('Sentiment').count()
            totale = len(df_load)

            plt.figure(figsize=(15, 15))
            plt.title("Numerosità Classi Dataset Testi: Totale " + str(totale) + " Elementi")
            graph = sns.barplot(x=gb.index, y=gb['OriginalTweet'].values)
            i = 0
            for p in graph.patches:
                height = p.get_height()
                graph.text(p.get_x() + p.get_width() / 2., height + 0.3, gb['OriginalTweet'].values[i], ha="center")
                i += 1
            plt.show()

    cv = TfidfVectorizer(max_features=max_features)
    enc = LabelEncoder()
    data = cv.fit_transform(df_load['StemmedString'])
    data = data.toarray()
    label = enc.fit_transform(df_load['Sentiment'])
    classes = enc.classes_

    if scale:
        s = StandardScaler()
        data = s.fit_transform(data)

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=split, random_state=SEED)

    if report:
        class_len = dict.fromkeys(classes)
        for k in classes:
            l = df_load[df_load['Sentiment'] == k]['StemmedTweet'].apply(len)
            class_len[k] = l
        mean_len = df_load['StemmedTweet'].apply(len)
        mean_len = np.round(mean_len.mean(), 1)

        i = 0
        color = ['r', 'g', 'b', 'c', 'y']
        num_plot = 5
        if reduce_labels:
            num_plot = 3
        fig = plt.figure(figsize=(15, 15))
        for k in classes:
            l = class_len[k]
            ax = fig.add_subplot(1, num_plot, i + 1)
            ax.hist(l, alpha=0.6, bins=20, color=color[i], label=k)
            i += 1
            ax.legend()
            ax.set_xlabel('Numero di Token')
            ax.set_ylabel('Numero di Tweet')
        fig.suptitle('Numero di Token per Tweet in base alla Classe')
        fig.tight_layout()

        print(f'Numero medio di token di cui si compongono i tweet: {mean_len}')
        print(f'Dimensione del dataset ottenuto (X, Y): {data.shape}, {label.shape}')
        print(f'Dimensione training/test set: {x_train.shape}, {x_test.shape}; split: {split}')
        print(f'Classi: {classes, enc.transform(classes)}')

    if return_other:
        return x_train, x_test, y_train, y_test, classes, cv
    else:
        return x_train, x_test, y_train, y_test


def train_test_load_intindex(max_features=None, sequence_len=500, split=.25, reduce_labels=False, report=False, return_other=False, scale=False):

    exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'))
    if exist:
        with open(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'), 'rb') as hf:
            df_load = pickle.load(hf)
    else:
        raise RuntimeError('Missing text_dataframe_preproc file, please run first "data_preprocessing.py" script to obtain this file.')

    # TERMINO LE OPERAZIONI DI PREPROCESSING
    if reduce_labels:
        df_load['Sentiment'] = df_load['Sentiment'].replace(['Extremely Negative', 'Extremely Positive'], ['Negative', 'Positive'])

        if report:
            gb = df_load.groupby('Sentiment').count()
            totale = len(df_load)

            plt.figure(figsize=(15, 15))
            plt.title("Numerosità Classi Dataset Testi: Totale " + str(totale) + " Elementi")
            graph = sns.barplot(x=gb.index, y=gb['OriginalTweet'].values)
            i = 0
            for p in graph.patches:
                height = p.get_height()
                graph.text(p.get_x() + p.get_width() / 2., height + 0.3, gb['OriginalTweet'].values[i], ha="center")
                i += 1
            plt.show()

    enc = LabelEncoder()
    Y = enc.fit_transform(df_load['Sentiment'])
    classes = enc.classes_

    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=max_features,  # NUMERO TOTALE DELLE PAROLE DEL VOCABOLARIO DA CONSIDERARE, SI PRENDONO LE TOP 'max_features' PER OCCORRENZA
        output_mode="int",
        output_sequence_length=sequence_len,  # VALORE CHE VA AD UNIFORMARE LA LUNGHEZZA DEI RECORD DEL DATASET DOPO LE OPERAZIONI DI TRASFORMAZIONE
    )

    data = df_load['StemmedString'].values
    vectorize_layer.adapt(data)
    vocabulary = vectorize_layer.get_vocabulary()

    X = vectorize_layer(data)
    X = X.numpy()

    if scale: # VADO AD EFFETTUARE SCALING NEL CASO IN CUI NON IMPIEGO RETI NEURALI PERCHE' NON HO A DISPOPIZIONE IL LAYER DI EMBEDDING CHE B
        s = StandardScaler()
        X = s.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=SEED, stratify=Y)

    if report:

        class_len = dict.fromkeys(classes)
        for k in classes:
            l = df_load[df_load['Sentiment'] == k]['StemmedTweet'].apply(len)
            class_len[k] = l
        mean_len = df_load['StemmedTweet'].apply(len)
        mean_len = np.round(mean_len.mean(), 1)

        i = 0
        color = ['r', 'g', 'b', 'c', 'y']
        num_plot = 5
        if reduce_labels:
            num_plot = 3
        fig = plt.figure(figsize=(15, 15))
        for k in classes:
            l = class_len[k]
            ax = fig.add_subplot(1, num_plot, i + 1)
            ax.hist(l, alpha=0.6, bins=20, color=color[i], label=k)
            i += 1
            ax.legend()
            ax.set_xlabel('Numero di Token')
            ax.set_ylabel('Numero di Tweet')
        fig.suptitle('Numero di Token per Tweet in base alla Classe')
        fig.tight_layout()

        print(f'Numero medio di token di cui si compongono i tweet: {mean_len}')
        print(f'Dimensione del dataset ottenuto (X, Y): {X.shape}, {Y.shape}')
        print(f'Dimensione training/test set: {x_train.shape}, {x_test.shape}; split: {split}')
        print(f'Classi: {classes, enc.transform(classes)}')

    if return_other:
        return x_train, x_test, y_train, y_test, classes, vocabulary
    else:
        return x_train, x_test, y_train, y_test


def load_dataset_anomaly_detection(mode, max_features=None, sequence_len=100, split=.25, return_other=False, plot=True):
    exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'))
    if exist:
        with open(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'), 'rb') as hf:
            df = pickle.load(hf)
    else:
        raise RuntimeError(
            'Missing text_dataframe_preproc file, please run first "data_preprocessing.py" script to obtain this file.')

    # VADO A RAFFINARE IL PREPROCESSING DEI DATI PER I MODELLI NON BASATI SU RETI NEURALI
    df['Sentiment'] = df['Sentiment'].replace(['Extremely Negative', 'Extremely Positive'], ['Negative', 'Positive']) # POSITIVE CLASSE NORMALE PERCHE' LA PIU' NUMEROSA
    df['Anomaly'] = df['Sentiment'].replace(['Negative', 'Neutral', 'Positive'], [1, 1, 0])
    df['Anomaly'] = df['Anomaly'].astype('int')
    token_len = df['StemmedTweet'].apply(len)
    df[token_len <= 2] = np.nan
    df = df.dropna()
    df = df.drop_duplicates(subset=['StemmedString'])

    # EFFETTUO ALCUNI PLOT
    if plot:
        gb_1 = df.groupby('Sentiment').count()['OriginalTweet']
        gb_2 = df.groupby('Anomaly').count()['OriginalTweet']

        fig = plt.figure(figsize=(15, 15))
        fig.suptitle("Anomaly Detection\nNumerosità Classi Dataset Testi: Totale " + str(len(df)) + " Elementi")
        fig.tight_layout()

        ax1 = fig.add_subplot(1, 2, 1)
        graph = sns.barplot(x=gb_1.index.to_numpy(), y=gb_1.values, ax=ax1)
        i = 0
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height + 0.1, gb_1.values[i], ha="center")
            i += 1

        ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
        graph = sns.barplot(x=['Normal Class ("Positive")', 'Anomaly Class ("Negative", "Neutral")'], y=gb_2.values, ax=ax2)
        i = 0
        for p in graph.patches:
            height = p.get_height()
            graph.text(p.get_x() + p.get_width() / 2., height + 0.1, gb_2.values[i], ha="center")
            i += 1
        plt.show()

    Y = df['Anomaly'].to_numpy()
    Y = Y.astype('int')

    if mode == 'int':
        vectorize_layer = TextVectorization(
            standardize=None,
            max_tokens=max_features,
            output_mode="int",
            output_sequence_length=sequence_len,
        )

        data = df['StemmedString'].values
        vectorize_layer.adapt(data)
        vocabulary = vectorize_layer.get_vocabulary()

        X = vectorize_layer(data)
        X = X.numpy()
        X_normal = X[Y == 0]
        x_train, x_test = train_test_split(X_normal, test_size=split, random_state=SEED)

        print(f'Mode: {mode}')
        print(f'Full dataset size: {X.shape}')
        print(f'Normal class dataset size: {X_normal.shape}')
        print(f'Train/Test Set size: {x_train.shape}/{x_test.shape}; (split = {split}) \n')

        if return_other:
            return x_train, x_test, X, Y, vocabulary
        else:
            return x_train, x_test, X, Y

    elif mode == 'tfidf': # PER QUESTA MODE NON VIENE USATO IL PARAMETRO 'sequence_len'
        vectorize_layer = TfidfVectorizer(max_features=max_features)
        data = vectorize_layer.fit_transform(df['StemmedString'])
        X = data.toarray()
        X_normal = X[Y == 0]
        x_train, x_test = train_test_split(X_normal, test_size=split, random_state=SEED)

        print(f'Mode: {mode}')
        print(f'Full dataset size: {X.shape}')
        print(f'Normal class dataset size: {X_normal.shape}')
        print(f'Train/Test Set size: {x_train.shape}/{x_test.shape}; (split = {split}) \n')

        if return_other:
            return x_train, x_test, X, Y, vectorize_layer
        else:
            return x_train, x_test, X, Y

    else:
        raise RuntimeError('Unsupported mode operation!')