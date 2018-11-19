1. introduction of scripts
	* ds1_model.py  : to generate model(.joblib), generate xxxVal.csv, grid search for dataset1 
	* ds2_model.py  : to generate model(.joblib), generate xxxVal.csv, grid search for dataset2
	* classifier.py  : to use model(.joblib) and generate xxxTest-dt.csv for both dataset1 and dataset2


2. environment requirement
	* python3 
	* scikit-learn 0.20.0
	* numpy 1.14.0


3. to run ds1_model.py / ds2_model.py
	* set up the dataset path(at row 100)
		```
		X_train, Y_train = parse_instances_vector_label('test/ds1/ds1Train.csv')
		X_validate, Y_validate = parse_instances_vector_label('test/ds1/ds1Val.csv')
		```

4. to run classifier.py
	* set up file_prefix and testdataset path(row 79/97)
		```
		file_prefix = 'ds1'
		X_test = parse_instance_vector('test/ds1/ds1Test.csv')
		```