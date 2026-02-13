import urllib.request
import pandas as pd
import requests
import io
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

link = "https://stats.idre.ucla.edu/stat/data/hsb2.csv"
webUrl = urllib.request.urlopen(link)
if webUrl.getcode() == 200:
    print("URL read successfully")
else:
    print("URL not read successfully, check url link/internet connection and try again")

s = requests.get(link).content
hsb2 = pd.read_csv(io.StringIO(s.decode('utf-8')))

#hsb2.head()
#hsb2.dtypes

hsb2["race"] = hsb2["race"].astype('category')
hsb2["female"] = hsb2["female"].astype('category')
hsb2["ses"] = hsb2["ses"].astype('category')

#hsb2.dtypes

race = pd.get_dummies(hsb2['race'],drop_first=True,prefix='race')
ses = pd.get_dummies(hsb2['ses'],drop_first=True,prefix='ses')
hsb3 = hsb2
hsb3.drop(['race','ses'],axis=1,inplace=True)
hsb3 = pd.concat([hsb3,race,ses],axis=1)

y = hsb3['prog']
X = hsb3.drop(['prog','id'],axis=1)

# define the multinomial logistic regression model with a default penalty
model = LogisticRegression(solver='lbfgs', penalty='l2', 
                           C=1.0, max_iter = 1000000)
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

result = model.fit(X, y)

row = X.iloc[0:1, :]
# predict a multinomial probability distribution
yhat = model.predict_proba(row)
# summarize the predicted probabilities
print('Predicted Probabilities: %s' % yhat[0])

# predict the class label
yhat = model.predict(row)
# summarize the predicted class
print('Predicted Class: %d' % yhat[0])

'''
# get a list of models to evaluate
def get_models():
	models = dict()
	for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
		# create name for model
		key = '%.4f' % p
		# turn off penalty in some cases
		if p == 0.0:
			# no penalty in this case
			models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
		else:
			models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()

for name, model in models.items():
	# evaluate the model and collect the scores
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize progress along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show();

print(result.intercept_)
print(result.coef_)

summary = pd.DataFrame(zip(X.columns, np.transpose(result.coef_.tolist()[0])), 
                       columns=['features', 'coef'])

print(summary)

'''

