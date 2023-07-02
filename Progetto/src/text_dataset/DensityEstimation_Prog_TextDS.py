import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from src.path_folders import DRIVE_FOLDER

from load_text_dataset import train_test_load_tfidf

def report_plot(r_df, title):
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    w = 0.3
    x_ticks = np.arange(len(classes))
    r1 = axis.bar(x_ticks-w, r_df['precision'].values[0:len(classes)], width=w, color='b', align='center', label='Precision')
    r2 = axis.bar(x_ticks, r_df['recall'].values[0:len(classes)], width=w, color='g', align='center', label='Recall')
    r3 = axis.bar(x_ticks+w, r_df['f1-score'].values[0:len(classes)], width=w, color='r', align='center', label='F1-Score')
    axis.set_ylabel('Scores')
    axis.set_title(title+' Classification Report \nTest Set Accuracy: '+str(np.round(r_df.loc['accuracy'][0], 2)))
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(classes)
    axis.legend(loc='upper right')
    axis.bar_label(r1, padding=3)
    axis.bar_label(r2, padding=3)
    axis.bar_label(r3, padding=3)
    fig.tight_layout()
    plt.show()

#CARICAMENTO DEI DATASET

SEED = 4040
np.random.seed(SEED)

max_tokens = 500 # NUMERO MASSIMO DI TOKEN CHE SI CONSIDERANO IN BASE ALLE LORO OCCORRENZE NEI RECORD
split = .15

x_train, x_test, y_train, y_test, classes, vocabulary = train_test_load_tfidf(max_features=max_tokens,  split=split, reduce_labels=True, report=True, return_other=True)

# VAD0 A TESTARE DUE MODELLI ANDANDO AD EFFETTUARE PARAMETER TUNING TRAMITE GRIDSEARCH CV
# 1) KNN:

CV = 5

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'knn_grid_search_text.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'knn_grid_search_text.hist'), 'rb') as hist_file:
        knn_grid_search_CV = pickle.load(hist_file)
else:

    knn = KNeighborsClassifier()

    params = {
        'n_neighbors': [3, 5, 10, 20],
        'weights': ['uniform', 'distance']
    }

    knn_grid_search_CV = GridSearchCV(knn, params, cv=CV, scoring='accuracy', n_jobs=-1, verbose=3)

    knn_grid_search_CV.fit(x_train, y_train)

    with open(os.path.join(DRIVE_FOLDER, 'knn_grid_search_text.hist'), 'wb') as hist_file:
        pickle.dump(knn_grid_search_CV, hist_file)

best_knn = knn_grid_search_CV.best_estimator_
best_val_accuracy = np.round(knn_grid_search_CV.best_score_, 3)

print('KNN Classifier:')
print(f'Miglior estimator ottenuto: {best_knn}; K={best_knn.n_neighbors}')
print(f'Valore medio di validation accuracy sui {CV} fold sviluppati da parte del miglior estimator: {best_val_accuracy} \n')

# VADO A TESTARE L'ESTIMATOR OTTENUTO MEDIANTE 10 FOLD CV

FOLDS = 10
VERBOSE = 1

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'best_knn_cv_text.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'best_knn_cv_text.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(best_knn, x_train, y_train, cv=kfold, verbose=VERBOSE, n_jobs=-1)
    with open(os.path.join(DRIVE_FOLDER, 'best_knn_cv_text.hist'), 'wb') as hist_file:
        pickle.dump(cv_results, hist_file)

# VADO A PLOTTARE I RISULTATI OTTENUTI

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
plt.title(f'Density Estimation (KNN) \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# 2) BERNOULLI NAIVE BAYES

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'bnaiveb_grid_search_text.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'bnaiveb_grid_search_text.hist'), 'rb') as hist_file:
        naiveb_grid_search_CV = pickle.load(hist_file)
else:

    naiveb = BernoulliNB()

    # CALCOLO LE PROBABILITA' APRIORI COME LA PERCENTUALE DI ELEMENTI APPARTENENTI ALLA DATA CLASSE NEL DATASET DI TRAIN AL FINE DI PROVARE TALE IPERPARAMETRO NEL GRIDS
    unique = np.unique(y_train)
    p = np.ones((unique.size, ))
    for i in range(len(unique)):
        elem = unique[i]
        count = (y_train == elem).sum()
        weight = count / y_train.size
        p[i] = weight

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1., 1.5, 2.],
        'fit_prior': [True, False]
    }

    naiveb_grid_search_CV = GridSearchCV(naiveb, params, cv=CV, scoring='accuracy', n_jobs=-1, verbose=3)

    naiveb_grid_search_CV.fit(x_train, y_train)

    with open(os.path.join(DRIVE_FOLDER, 'bnaiveb_grid_search_text.hist'), 'wb') as hist_file:
        pickle.dump(naiveb_grid_search_CV, hist_file)

best_naiveb = naiveb_grid_search_CV.best_estimator_
best_val_accuracy_nb = np.round(naiveb_grid_search_CV.best_score_, 3)

print('Bernoulli Naive Bayes Classifier:')
print(f'Miglior estimator ottenuto: {best_naiveb}')
print(f'Valore medio di validation accuracy sui {CV} fold sviluppati da parte del miglior estimator: {best_val_accuracy_nb} \n')

# VADO A TESTARE L'ESTIMATOR OTTENUTO MEDIANTE 10 FOLD CV

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'best_bnaiveb_cv_text.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'best_bnaiveb_cv_text.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(best_naiveb, x_train, y_train, cv=kfold, verbose=VERBOSE, n_jobs=-1)
    with open(os.path.join(DRIVE_FOLDER, 'best_bnaiveb_cv_text.hist'), 'wb') as hist_file:
        pickle.dump(cv_results, hist_file)

# VADO A PLOTTARE I RISULTATI OTTENUTI

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
plt.title(f'Density Estimation (Bernoulli Naive Bayes) \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# SCELGO DI IMPIEGARE IL MODELLO BERNOULLI NAIVE BAYES PERCHE' MI RESTITUISCE I RISULTATI MIGLIORI
# VADO AD EFFETTUARE IL TESTING DEL MODELLO SCELTO E CALCOLO IL CLASSIFICATION REPORT

predictions = best_naiveb.predict(x_test)

report = classification_report(y_test, predictions, target_names=classes, output_dict=False, zero_division=0)
report_dict = classification_report(y_test, predictions, target_names=classes, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()

print("Classification Report:")
print(report)

# VADO A PLOTTARE I RISULTATI

report_plot(report_df, 'Density Estimation (Bernoulli Naive Bayes)')