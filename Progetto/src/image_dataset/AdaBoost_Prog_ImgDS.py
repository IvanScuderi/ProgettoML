import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from load_img_dataset import load_img_dataset
from src.path_folders import DRIVE_FOLDER


def report_plot(r_df):
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(111)
    w = 0.3
    x_ticks = np.arange(4)
    r1 = axis.bar(x_ticks-w, r_df['precision'].values[0:4], width=w, color='b', align='center', label='Precision')
    r2 = axis.bar(x_ticks, r_df['recall'].values[0:4], width=w, color='g', align='center', label='Recall')
    r3 = axis.bar(x_ticks+w, r_df['f1-score'].values[0:4], width=w, color='r', align='center', label='F1-Score')
    axis.set_ylabel('Scores')
    axis.set_title('AdaBoost Classification Report \n Test Set Accuracy: '+str(np.round(r_df.loc['accuracy'][0], 4)))
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

# VADO A DEFINIRE IL CLASSIFICATORE ADABOOST IMPIEGANDO GRID SEARCH CV CON 5 FOLD PER TROVARE IL MIGLIOR BASE ESTIMATOR E IL MIGLIOR NUMERO DI ESTIMATOR

CV = 5

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'ada_grid_search.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'ada_grid_search.hist'), 'rb') as hist_file:
        ada_grid_search_CV = pickle.load(hist_file)
else:

    ada = AdaBoostClassifier(random_state=SEED)

    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'learning_rate': [.7, 1, 1.5, 2, 3, 5]
    }

    ada_grid_search_CV = GridSearchCV(ada, params, cv=CV, scoring='accuracy', n_jobs=-1, verbose=3)

    ada_grid_search_CV.fit(x_train, y_train)

    with open(os.path.join(DRIVE_FOLDER, 'ada_grid_search.hist'), 'wb') as hist_file:
        pickle.dump(ada_grid_search_CV, hist_file)

best_ada = ada_grid_search_CV.best_estimator_
best_val_accuracy = np.round(ada_grid_search_CV.best_score_, 3)

print(f'Miglior estimator ottenuto: {best_ada}')
print(f'Valore medio di validation accuracy sui {CV} fold sviluppati da parte del miglior estimator: {best_val_accuracy} \n')

# VADO A TESTARE L'ESTIMATOR OTTENUTO MEDIANTE 10 FOLD CV

FOLDS = 10
VERBOSE = 1

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'best_ada_cv.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'best_ada_cv.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(best_ada, x_train, y_train, cv=kfold, verbose=VERBOSE, n_jobs=-1)
    with open(os.path.join(DRIVE_FOLDER, 'best_ada_cv.hist'), 'wb') as hist_file:
        pickle.dump(cv_results, hist_file)

# VADO A PLOTTARE I RISULTATI OTTENUTI

plt.figure(figsize=(10, 10))
plt.ylim([0, 1])
plt.plot(range(1, 11), cv_results, 'ro', range(1, 11), cv_results, 'k--')
plt.title(f'AdaBoost \n Validation Accuracy Durante i 10 Folds  \n Validation Accuracy media: {np.round(cv_results.mean(),4)}')
plt.xlabel('Folds')
plt.xticks(range(1, 11))
plt.ylabel('Validation Accuracy')
plt.show()

# VADO A STAMPARE UN CLASSIFICATION REPORT SUL DATASET DI TEST

predictions = best_ada.predict(x_test)

report = classification_report(y_test, predictions, target_names=class_names, output_dict=False)
report_dict = classification_report(y_test, predictions, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

print("Classification Report:")
print(report)

# VADO A PLOTTARE I RISULTATI

report_plot(report_df)














