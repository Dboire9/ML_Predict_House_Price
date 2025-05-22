import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


def main():

	# Importing the data and cleaning it
	
	train = pd.read_csv('data/train.csv')
	test = pd.read_csv('data/test.csv')
	
	print(train.shape)
	print(test.shape)
	
	plt.style.use(style='ggplot')
	plt.rcParams['figure.figsize'] = (10, 6)

	# Reducing the skewness
	target = np.log(train.SalePrice)

	# Getting only numeric data types for now
	numeric_features = train.select_dtypes(include=[np.number])

	# Showing the positive and negative correlation to the SalePrice
	corr = numeric_features.corr()
	print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
	print (corr['SalePrice'].sort_values(ascending=False)[-5:])

	# Quality pivot to see relationship between OverallQual (#1 in positive correlation) and SalePrice
	quality_pivot = train.pivot_table(index='OverallQual',
					values='SalePrice', aggfunc=np.median)
	# Because it has values between 0 and 10 we can plot it with bars
	quality_pivot.plot(kind='bar', color='blue')
	plt.xlabel('Overall Quality')
	plt.ylabel('Median Sale Price')
	plt.show()
	
	# Reducing the outliers
	train = train[train['GarageArea'] < 1200]

	# Plotting the #2 best positive correlation, because it has many unique data, we scatter it
	plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
	plt.ylabel('Sale Price')
	plt.xlabel('Garage Area')
	plt.show()

	# Checking the null values in our data set
	nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
	nulls.columns = ['Null Count']
	nulls.index.name = 'Feature'
	# print(nulls)

	# Checking the non numeric data types
	categoricals = train.select_dtypes(exclude=[np.number])
	# print(categoricals.describe())

	# Feature engineering the Street column
	# We can see that the Street column has only two values, Grvl and Pave
	train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
	test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
	
	# Interpolate the missing values to fill the gaps in our data
	data = train.select_dtypes(include=[np.number]).interpolate().dropna()
	
	# Checking the null values in our data set
	print(sum(data.isnull().sum() != 0))
	
	# Beginning to build the model
	
	y = np.log(train.SalePrice)
	# We drop the SalePrice and Id columns from the train data set
	X = data.drop(['SalePrice', 'Id'], axis=1)
	
	# We put test_size at 0.33 so we have 0.67 percent for training and 0.33 for validation
	# We use random_state to make the results reproducible
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	# Initializing the model
	lr = linear_model.LinearRegression()
	model = lr.fit(X_train, y_train)
	
	# Kaggle will evaluate the model with the R^2 score
	print ("R^2 is: \n", model.score(X_test, y_test))
	
	# Check if the model is good 
	predictions = model.predict(X_test)
	print ('RMSE is: \n', mean_squared_error(y_test, predictions))
	
	
	# Plotting the predicted values against the actual values to see how well the model is doing
	actual_values = y_test
	plt.scatter(predictions, actual_values, alpha=.7,
				color='b')
	plt.xlabel('Predicted Price')
	plt.ylabel('Actual Price')
	plt.title('Linear Regression Model')
	plt.show()
	
	# Making the submission file
	submission = pd.DataFrame()
	submission['Id'] = test.Id
	feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
	
	# Makking the predictions
	predictions = model.predict(feats)
	# Reversing the log transformation
	final_predictions = np.exp(predictions)
	
	# Putting the predictions in the submission file
	submission['SalePrice'] = final_predictions
	
	# Checking the submission file
	print(submission.head())
	
	# Making the submission file
	submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
	main()