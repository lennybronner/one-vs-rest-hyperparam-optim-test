from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import scipy

from my_one_vs_rest import MyOneVsRestClassifier

def create_classifier(hyperparameters, method, n=-1):
    if method == 'sklearn':    
        if hyperparameters is not None and 'estimator__alpha' in hyperparameters:
            base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet', tol=1e-3, alpha=hyperparameters['estimator__alpha'])
        else:
            base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet', tol=1e-3)
        return OneVsRestClassifier(base_classifier, n_jobs=-1)
    elif method == 'own':
        if hyperparameters is not None:
            base_classifiers = [SGDClassifier(loss='modified_huber', penalty='elasticnet', tol=1e-3, alpha=hyperparameters[i]['alpha']) for i in range(n)]
        else:
            base_classifiers = [SGDClassifier(loss='modified_huber', penalty='elasticnet', tol=1e-3) for _ in range(n)]
        return MyOneVsRestClassifier(base_classifiers, n_jobs=-1)

def find_best_hyperparmeters(dtm, labels, method, n=-1):
    classifier = create_classifier(None, method, n)
    if method == 'sklearn':
        param_distribution = {'estimator__alpha': scipy.stats.expon(scale=0.00001)}
        extra_kwargs = {'n_jobs': -1}
        scv = RandomizedSearchCV(classifier, param_distribution, n_iter=20, cv=3, scoring='f1_micro', iid=True, verbose=1, refit=False, **extra_kwargs)
        scv.fit(dtm, labels)
        return scv.best_params_
    elif method == 'own':
        label_binarizer = LabelBinarizer(sparse_output=True)
        Y = label_binarizer.fit_transform(labels)
        Y = Y.tocsc()
        classes_ = label_binarizer.classes_
        best_params = []
        columns = list(col.toarray().ravel() for col in Y.T)
        for i, column in enumerate(columns):
            param_distribution = {'alpha': scipy.stats.expon(scale=0.00001)}
            extra_kwargs = {'n_jobs': -1}
            scv = RandomizedSearchCV(classifier.estimators[i], param_distribution, n_iter=20, cv=3, scoring='f1_micro', iid=True, verbose=1, refit=False, **extra_kwargs)
            if np.sum(column) > 1:
                scv.fit(dtm, columns[i])
                best_params.append(scv.best_params_)
            else:
                best_params.append({'alpha': 0.0001}) # default
        return best_params
            

def kfold_cross_validate(dtm, labels, hyperparameters, method, n=-1):
    f1s, precisions, recalls = [], [], []
    kf = KFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(kf.split(dtm)):
        classifier = create_classifier(hyperparameters, method, n)
        X_train, X_test = dtm[train_index], dtm[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average='micro'))
        precisions.append(precision_score(y_test, y_pred, average='micro'))
        recalls.append(recall_score(y_test, y_pred, average='micro'))

    print(f"precision: {np.mean(precisions)}")
    print(f"recall: {np.mean(recalls)}")
    print(f"f1: {np.mean(f1s)}")
