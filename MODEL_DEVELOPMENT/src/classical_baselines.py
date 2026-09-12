import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def train_eval_logistic_regression(X_train, y_train, X_val):
    n_comp = min(16, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    clf = LogisticRegression(solver='liblinear', max_iter=200, C=1.0, random_state=42)
    clf.fit(X_train_pca, y_train)
    y_prob = clf.predict_proba(X_val_pca)[:, 1]
    return clf, y_prob

def train_eval_svm(X_train, y_train, X_val):
    n_comp = min(16, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    # Fast SGDClassifier with log_loss for SVM/linear probability calibration
    clf = SGDClassifier(loss='log_loss', penalty='l2', max_iter=200, random_state=42)
    clf.fit(X_train_pca, y_train)
    y_prob = clf.predict_proba(X_val_pca)[:, 1]
    return clf, y_prob

def train_eval_random_forest(X_train, y_train, X_val):
    n_comp = min(16, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    clf.fit(X_train_pca, y_train)
    y_prob = clf.predict_proba(X_val_pca)[:, 1]
    return clf, y_prob
