import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from load_img_dataset import load_img_dataset
from src.path_folders import DRIVE_FOLDER

def report_plot(r_df, alg):
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    w = 0.3
    x_ticks = np.arange(4)
    r1 = axis.bar(x_ticks-w, r_df['precision'].values[0:4], width=w, color='b', align='center', label='Precision')
    r2 = axis.bar(x_ticks, r_df['recall'].values[0:4], width=w, color='g', align='center', label='Recall')
    r3 = axis.bar(x_ticks+w, r_df['f1-score'].values[0:4], width=w, color='r', align='center', label='F1-Score')
    axis.set_ylabel('Scores')
    axis.set_title(f'Density Estimation ({alg}) Classification Report \n Test Set Accuracy: '+str(np.round(r_df.loc['accuracy'][0], 4)))
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(class_names)
    axis.legend(loc='upper right')
    axis.bar_label(r1, padding=3)
    axis.bar_label(r2, padding=3)
    axis.bar_label(r3, padding=3)
    fig.tight_layout()
    plt.show()

SEED = 4040
split = .2
image_size = (60, 80)
class_names = ['Shirts', 'Shorts', 'Sunglasses', 'Wallets']

# VADO A CARICARE IL DATASET DI IMMAGINI EFFETTUANDO LO SPLIT IN TRAIN E TEST SET

x_train_rgb, x_test_rgb, y_train, y_test = load_img_dataset(split=split, report=True)

# COME PRIMA COSA SICCOME GLI ESTIMATOR DI SKLEARN NECESSITANO DI NDARRAY IN 2 DIM PORTO LE IMG DEL DATASET IN SCALA DI GRIGI

tmp_train = []
for img in x_train_rgb:
    gray_img = rgb2gray(img)
    tmp_train.append(gray_img)

tmp_train = np.asarray(tmp_train)
tmp_train = tmp_train.reshape(tmp_train.shape[0], 60*80)

tmp_test = []
for img in x_test_rgb:
    gray_img = rgb2gray(img)
    tmp_test.append(gray_img)

tmp_test = np.asarray(tmp_test)
tmp_test = tmp_test.reshape(tmp_test.shape[0], 60*80)

x_train, x_test = tmp_train, tmp_test

print(f'Dimensione Training / Test Set post conversione immagini in scala di grigi: {x_train.shape}, {x_test.shape} \n')

# VADO A PLOTTARE ALCUNE IMMAGINI PER CONTROLLARE IL COMPLETAMENTO CORRETTO DELL'OPERAZIONE

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape(image_size), cmap='gray')
    plt.xlabel(class_names[y_train[i]])
plt.show()

# SCELTA PROGETTUALE: UTILIZZO IL KNN CLASSIFIER

CV = 10

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'knn_grid_search.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'knn_grid_search.hist'), 'rb') as hist_file:
        knn_grid_search_CV = pickle.load(hist_file)
else:

    knn = KNeighborsClassifier()

    params = {
        'n_neighbors': [3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3]
    }

    knn_grid_search_CV = GridSearchCV(knn, params, cv=CV, scoring='accuracy', n_jobs=-1, verbose=3)

    knn_grid_search_CV.fit(x_train, y_train)

    with open(os.path.join(DRIVE_FOLDER, 'knn_grid_search.hist'), 'wb') as hist_file:
        pickle.dump(knn_grid_search_CV, hist_file)

best_knn = knn_grid_search_CV.best_estimator_
best_val_accuracy = np.round(knn_grid_search_CV.best_score_, 3)

print('KNN Classifier:')
print(f'Miglior estimator ottenuto: {best_knn}')
print(f'Valore medio di validation accuracy sui {CV} fold sviluppati da parte del miglior estimator: {best_val_accuracy} \n')

# KNN: VADO A TESTARE L'ESTIMATOR OTTENUTO MEDIANTE 10 FOLD CV

FOLDS = 10
VERBOSE = 1

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'best_knn_cv.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'best_knn_cv.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(best_knn, x_train, y_train, cv=kfold, verbose=VERBOSE, n_jobs=-1)
    with open(os.path.join(DRIVE_FOLDER, 'best_knn_cv.hist'), 'wb') as hist_file:
        pickle.dump(cv_results, hist_file)

# KNN: VADO A PLOTTARE I RISULTATI OTTENUTI

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
plt.title(f'Density Estimation (KNN) \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# VADO ORA A PROVARE UN NAIVE BAYES CLASSIFIER

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'naiveb_grid_search.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'naiveb_grid_search.hist'), 'rb') as hist_file:
        naiveb_grid_search_CV = pickle.load(hist_file)
else:

    naiveb = GaussianNB()

    # CALCOLO LE PROBABILITA' APRIORI COME LA PERCENTUALE DI ELEMENTI APPARTENENTI ALLA DATA CLASSE NEL DATASET DI TRAIN AL FINE DI PROVARE TALE IPERPARAMETRO NEL GRIDS
    unique = np.unique(y_train)
    p = np.ones((unique.size, ))
    for i in range(len(unique)):
        elem = unique[i]
        count = (y_train == elem).sum()
        weight = count / y_train.size
        p[i] = weight

    params = {
        'var_smoothing': [1e-5, 1e-7, 1e-8, 1e-9, 1e-10],
        'priors': [None, p]
    }

    naiveb_grid_search_CV = GridSearchCV(naiveb, params, cv=CV, scoring='accuracy', n_jobs=-1, verbose=3)

    naiveb_grid_search_CV.fit(x_train, y_train)

    with open(os.path.join(DRIVE_FOLDER, 'naiveb_grid_search.hist'), 'wb') as hist_file:
        pickle.dump(naiveb_grid_search_CV, hist_file)

best_naiveb = naiveb_grid_search_CV.best_estimator_
best_val_accuracy_nb = np.round(naiveb_grid_search_CV.best_score_, 3)

print('Naive Bayes Classifier:')
print(f'Miglior estimator ottenuto: {best_naiveb}')
print(f'Valore medio di validation accuracy sui {CV} fold sviluppati da parte del miglior estimator: {best_val_accuracy_nb} \n')

# NAIVE BAYES: VADO A TESTARE L'ESTIMATOR OTTENUTO MEDIANTE 10 FOLD CV

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'best_naiveb_cv.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'best_naiveb_cv.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(best_naiveb, x_train, y_train, cv=kfold, verbose=VERBOSE, n_jobs=-1)
    with open(os.path.join(DRIVE_FOLDER, 'best_naiveb_cv.hist'), 'wb') as hist_file:
        pickle.dump(cv_results, hist_file)

# NAIVE BAYES: VADO A PLOTTARE I RISULTATI OTTENUTI

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
plt.title(f'Density Estimation (Naive Bayes) \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# VADO A STAMPARE UN CLASSIFICATION REPORT SUL DATASET DI TEST
# KNN:

predictions = best_knn.predict(x_test)

report = classification_report(y_test, predictions, target_names=class_names, output_dict=False)
report_dict = classification_report(y_test, predictions, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("Classification Report (KNN):")
print(report)

# NAIVE BAYES:

predictions_nb = best_naiveb.predict(x_test)

report_nb = classification_report(y_test, predictions_nb, target_names=class_names, output_dict=False)
report_dict_nb = classification_report(y_test, predictions_nb, target_names=class_names, output_dict=True)
report_df_nb = pd.DataFrame(report_dict_nb).transpose()

print("Classification Report (Naive Bayes):")
print(report_nb)

# VADO A PLOTTARE I RISULTATI

report_plot(report_df, 'KNN')
report_plot(report_df_nb, 'Naive Bayes')