import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from load_text_dataset import train_test_load_tfidf
from src.path_folders import DRIVE_FOLDER

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

# VADO A DEFINIRE IL CLASSIFICATORE ADABOOST IMPIEGANDO GRID SEARCH CV CON 5 FOLD PER TROVARE IL MIGLIOR BASE ESTIMATOR E IL MIGLIOR NUMERO DI ESTIMATOR

CV = 5

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'ada_grid_search_text.hist'))
if exists:
    with open(os.path.join(DRIVE_FOLDER, 'ada_grid_search_text.hist'), 'rb') as hist_file:
        ada_grid_search_CV = pickle.load(hist_file)
else:

    ada = AdaBoostClassifier(random_state=SEED)

    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'learning_rate': [.5, .7, 1, 1.5, 2]
    }

    ada_grid_search_CV = GridSearchCV(ada, params, cv=CV, scoring='accuracy', n_jobs=-1, verbose=3)

    ada_grid_search_CV.fit(x_train, y_train)

    with open(os.path.join(DRIVE_FOLDER, 'ada_grid_search_text.hist'), 'wb') as hist_file:
        pickle.dump(ada_grid_search_CV, hist_file)

best_ada = ada_grid_search_CV.best_estimator_
best_val_accuracy = np.round(ada_grid_search_CV.best_score_, 3)

print(f'Miglior estimator ottenuto: {best_ada}')
print(f'Valore medio di validation accuracy sui {CV} fold sviluppati da parte del miglior estimator: {best_val_accuracy} \n')

# VADO A TESTARE L'ESTIMATOR OTTENUTO MEDIANTE 10 FOLD CV

FOLDS = 10
VERBOSE = 1

exist = os.path.isfile(os.path.join(DRIVE_FOLDER, 'best_ada_cv_text.hist'))
if exist:
    with open(os.path.join(DRIVE_FOLDER, 'best_ada_cv_text.hist'), 'rb') as hist_file:
        cv_results = pickle.load(hist_file)
else:
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True)
    cv_results = cross_val_score(best_ada, x_train, y_train, cv=kfold, verbose=VERBOSE, n_jobs=-1)
    with open(os.path.join(DRIVE_FOLDER, 'best_ada_cv_text.hist'), 'wb') as hist_file:
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

report = classification_report(y_test, predictions, target_names=classes, output_dict=False, zero_division=0)
report_dict = classification_report(y_test, predictions, target_names=classes, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()

print("Classification Report:")
print(report)

# VADO A PLOTTARE I RISULTATI

report_plot(report_df, 'AdaBoost')

