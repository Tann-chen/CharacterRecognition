import os
import csv
import math
import numpy as np
from sklearn.externals import joblib


def parse_instance_vector(file_name):
	X_test = list()

	try:
		source_file = open(file_name, 'r')
		reader = csv.reader(source_file)
		for row in reader:
			features = np.array(row).astype(int)
			X_test.append(features)
	except csv.Error as e:
		source_file.close()

	return X_test



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
		line_no = 0
		for r in rows:
			line_no += 1
			line = list()
			line.append(line_no)
			line.append(r)
			writer.writerow(line)
	
	except csv.Error as e:
		target_file.close()


if __name__ == '__main__':
	#--------- ds1 ----------
	file_prefix = 'ds1'
	X_test = parse_instance_vector('test/ds1/ds1Test.csv')
	X_pooling_test = count_pooling(X_test, 5, 1)
	#
	nb_clf = joblib.load( file_prefix + '_nb_classifier.joblib')
	Y_pred = nb_clf.predict(X_test)
	csv_write(file_prefix + 'Test-nb.csv',Y_pred)
	#
	dt_clf = joblib.load( file_prefix + '_dt_classifier.joblib')
	Y_pred = dt_clf.predict(X_pooling_test)
	csv_write(file_prefix + 'Test-dt.csv',Y_pred)
	#
	svm_clf = joblib.load( file_prefix + '_svm_classifier.joblib')
	Y_pred = svm_clf.predict(X_pooling_test)
	csv_write(file_prefix + 'Test-3.csv',Y_pred)


	#--------- ds2 ----------
	file_prefix = 'ds2'
	X_test = parse_instance_vector('test/ds2/ds2Test.csv')
	#
	nb_clf = joblib.load( file_prefix + '_nb_classifier.joblib')
	Y_pred = nb_clf.predict(X_test)
	csv_write(file_prefix + 'Test-nb.csv',Y_pred)
	#
	dt_clf = joblib.load( file_prefix + '_dt_classifier.joblib')
	Y_pred = dt_clf.predict(X_pooling_test)
	csv_write(file_prefix + 'Test-dt.csv',Y_pred)
	#
	svm_clf = joblib.load( file_prefix + '_svm_classifier.joblib')
	Y_pred = svm_clf.predict(X_pooling_test)
	csv_write(file_prefix + 'Test-3.csv',Y_pred)



