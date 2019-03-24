

import warnings
warnings.filterwarnings('ignore', message='numpy.dtype size')

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def get_data():
	data_dir = os.path.join('.', 'data')
	df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
	df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
	X_train = df_train.drop(['ID_code', 'target'], axis=1).values
	y_train = df_train['target'].values
	X_test = df_test.drop('ID_code', axis=1).values
	return X_train, y_train, X_test

def train(X_tr, X_cv, y_tr, y_cv, options):
	print(options)
	model = RandomForestClassifier(**options)
	model.fit(X_tr, y_tr)
	p_tr = model.predict_proba(X_tr)
	p_tr = p_tr[:,1]
	tr_score = roc_auc_score(y_tr, p_tr)
	p_cv = model.predict_proba(X_cv)
	p_cv = p_cv[:,1]
	cv_score = roc_auc_score(y_cv, p_cv)
	print(f'TR ROC-AUC: {tr_score:.5f}\n\
			\rCV ROC-AUC: {cv_score:.5f}')
	filename = os.path.join('.', 'model.sav')
	pickle.dump(model, open(filename, 'wb'))
	return model

def main():
	X_train, y_train, X_test = get_data()
	X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=0.2)
	options = {'n_estimators': 10, 'criterion': 'gini',
			'max_depth': None, 'min_samples_split': 2,
			'min_samples_leaf': 1, 'max_features': 'sqrt',
			'max_leaf_nodes': None, 'min_impurity_decrease': 0,
			'bootstrap': True, 'oob_score': True,
			'n_jobs': -1, 'random_state': 0, 'verbose': 0}
	model = train(X_tr, X_cv, y_tr, y_cv, options)

def main_predict():
	_, _, X_test = get_data()
	filename = os.path.join('.', 'model.sav')
	loaded_model = pickle.load(open(filename, 'rb'))
	p_test = loaded_model.predict_proba(X_test)
	submission = pd.read_csv(os.path.join('.', 'data', 'sample_submission.csv'))
	submission['target'] = p_test
	submission.to_csv(os.path.join('.', 'subm.csv'), index=False)
	print(p_test)

if __name__ == '__main__':
	# main()
	main_predict()