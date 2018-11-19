import csv
import math
import numpy as np
from sklearn.externals import joblib

from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV



def parse_instances_vector_label(file_name):
	X = list()
	Y = list()
	try:
		source_file = open(file_name, 'r')
		reader = csv.reader(source_file)
		for row in reader:
			features = np.array(row[: -1]).astype(int)
			label = int(row[-1])
			X.append(features)
			Y.append(label)
	except csv.Error as e:
		source_file.close()

	return X, Y



def grid_search(X, Y, classifier, params):
	gsearch = GridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=5)
	gsearch.fit(X, Y)
	best_params = gsearch.best_estimator_.get_params()
	print("Best score : " + str(gsearch.best_score_))
	for p, v in best_params.items():
		print(p + " : " + str(v))



def count_pooling(X, step_size, pad_num):
	after_pooling = []
	axis_len = int(math.sqrt(len(X[0])))

	for x in X:
		reshaped = np.reshape(x, (axis_len, axis_len))
		# padding 
		padded_axis_len = math.ceil(axis_len / step_size)
		after_padding = [pad_num] * int(math.pow(padded_axis_len * step_size, 2))		
		after_padding = np.reshape(after_padding, (padded_axis_len * step_size, padded_axis_len * step_size))
		
		for r in range(0, axis_len):
			for c in range(0, axis_len):
				if reshaped[r][c] != pad_num:
					after_padding[r][c] = reshaped[r][c]

		# pooling
		pool_result = []
		for r in range(0, padded_axis_len):
			for c in range(0, padded_axis_len):
				flag = 0
				for x in range(r * step_size, r * step_size + step_size):
					for y in range(c * step_size, c * step_size + step_size):
						if after_padding[x][y] == 0:
							flag += 1

						#flag = min(flag , after_padding[x][y])

				pool_result.append(flag)

		after_pooling.append(pool_result)

	return after_pooling


def csv_write(output_file, rows):
	try:
		target_file = open(output_file, 'w', encoding='gb18030', newline='')
		writer = csv.writer(target_file)
		row_no = 0
		for r in rows:
			row_no += 1
			line = list()
			line.append(row_no)
			line.append(r)
			writer.writerow(line)
	
	except csv.Error as e:
		target_file.close()



if __name__ == '__main__':

	# ==========================  Dataset 1 =============================

	X_train, Y_train = parse_instances_vector_label('test/ds1/ds1Train.csv')
	X_validate, Y_validate = parse_instances_vector_label('test/ds1/ds1Val.csv')
	X_train_pooling = count_pooling(X_train, 5, 1)
	X_validate_pooling = count_pooling(X_validate, 5, 1)
	
	# ---- naive bayes -----
	# grid search
	# X = X_train + X_valt
	# Y = Y_train + Y_valt
	# params = {'alpha': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)}
	# nb_classifier = BernoulliNB(alpha=0.1)
	# grid_search(X, Y, nb_classifier, params)

	# run valdiate 
	nb_classifier = BernoulliNB(alpha=0.1)
	nb_classifier.fit(X_train, Y_train)
	Y_pred = nb_classifier.predict(X_validate)
	csv_write('ds1Val-nb.csv', Y_pred)
	print("naive bayes binary values - accuracy : " + str(accuracy_score(Y_validate, Y_pred)))
	joblib.dump(nb_classifier, 'ds1_nb_classifier.joblib')

	# # run pooling
	nbp_classifier = MultinomialNB(alpha=0.1)
	nbp_classifier.fit(X_train_pooling, Y_train)
	Y_pred = nbp_classifier.predict(X_validate_pooling)
	print("naive bayes pooling data - accuracy : " + str(accuracy_score(Y_validate, Y_pred)))

	# ---- decision tree -----
	# grid search 
	# X = X_train + X_valt
	# Y = Y_train + Y_valt
	# params = {'min_samples_split': (2, 4, 6, 8, 10)}
	# dt_classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
	# grid_search(X, Y, dt_classifier, params)

	# run validate
	dt_classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
	dt_classifier.fit(X_train, Y_train)
	Y_pred = dt_classifier.predict(X_validate)
	print("decision tree binary values - accuracy : " + str(accuracy_score(Y_validate, Y_pred)))

	# # run pooling 
	dtp_classifier = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
	dtp_classifier.fit(X_train_pooling, Y_train)
	Y_pred = dtp_classifier.predict(X_validate_pooling)
	csv_write('ds1Val-dt.csv', Y_pred)
	print("decision tree pooling values - accuracy : " + str(accuracy_score(Y_validate, Y_pred)))
	joblib.dump(dtp_classifier, 'ds1_dt_classifier.joblib')


	# # svm
	# run validate
	svmclf = OneVsRestClassifier(estimator=SVC(gamma='scale', random_state=0))
	svmclf.fit(X_train, Y_train)
	Y_pred = svmclf.predict(X_validate)
	print("svm binary values - accuracy : " + str(accuracy_score(Y_validate, Y_pred)))

	# run pooling
	svmclfp = OneVsRestClassifier(estimator=SVC(gamma='scale', random_state=0))
	svmclfp.fit(X_train_pooling, Y_train)
	Y_pred = svmclfp.predict(X_validate_pooling)
	csv_write('ds1Val-3.csv', Y_pred)
	print("svm pooling values - accuracy : " + str(accuracy_score(Y_validate, Y_pred)))
	joblib.dump(svmclfp, 'ds1_svm_classifier.joblib')









