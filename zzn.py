import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(path):
	return pd.read_csv(path)


def preprocessing(df):
	# remove '?' values
	for column in df.columns:
		if df[column].dtype.name == 'object':
			df = df[df[column] != '?']

	# map strings to IDs
	from sklearn import preprocessing

	nominal_columns = [column for column in df.columns if df[column].dtype.name == 'object']
	replace_maps = {}

	for column in nominal_columns:
		le = preprocessing.LabelEncoder()
		le.fit(df[column])
		replace_map = dict(zip(le.classes_, le.transform(le.classes_)))
		replace_maps[column] = replace_map

	return df.replace(replace_maps), replace_maps


def main():
	pd.set_option('display.max_columns', 50)

	df = load_data("data/userprofile.csv")
	df, value_maps = preprocessing(df)

	X = df[['ambience', 'religion', 'drink_level', 'personality']]
	Y = df.smoker

	xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=42)

	clf = RandomForestClassifier(random_state=0)
	clf.fit(xtrain, ytrain)

	predictions = clf.predict(xtest)
	errors = abs(predictions - ytest)
	print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
	#print(list((xtest, predictions, ytest))[:10])

if __name__ == "__main__":
	main()
