import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import sklearn.preprocessing
import os,re
import nltk
import argparse
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, cross_val_predict


def get_output(name):
	#Importing the dataset
	data = pd.read_csv('~/Downloads/labelled_bug.csv')

	#Finding the Class Distribution
	data.label_name.value_counts()

	#Creating a DataFrame where only Bug ID, Description and Label are kept from the original Data
	train = pd.DataFrame({'BugID':data['BugID'], 'Description':data['Description'], 'Label':data['label'],'Priority':data['label_name']})

	#Define X and y from the data as 'feature' and 'target' so that we can Vectorize
	X = train['Description']
	y = train['Label']


	#Creating Dictionary of Several Severity Level
	priority = {
    	0:"Low",
    	1:"Medium" ,
    	2:"High" ,
    	3:"Immediate" 
	}

	#Split X and Y into training and testing sets
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 1)

	#Use CountVectorizer to convert text into a matrix of token counts
	#Learn the vocabulaty of the training data and transform it into a 'document-term' matrix
	from sklearn.feature_extraction.text import CountVectorizer
	#instantiate the Vectorizer
	count_vect = CountVectorizer(analyzer = 'word',tokenizer = nltk.word_tokenize, stop_words = 'english', max_df = 0.4, min_df = 3,ngram_range=(1,2))

	#Learn training data Vocabulary, then use it to create document-term matrix
	X_train_dtm = count_vect.fit_transform(X_train)

	#examining the document-term matrix
	pd.DataFrame(X_train_dtm.toarray(), columns=count_vect.get_feature_names())

	#transform testing data into document-term matrix using the fitted Vocabulary.
	X_test_dtm = count_vect.transform(X_test)

	#Converting Occurence Count to Frequency count i.e. Weighted TF-IDF matrix
	from sklearn.feature_extraction.text import TfidfTransformer
	tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_dtm)
	X_train_tf = tf_transformer.transform(X_train_dtm)
	X_test_tf = tf_transformer.transform(X_test_dtm)

	#import and instantiate a Multinomial Naive Bayes Model
	from sklearn.naive_bayes import MultinomialNB
	nb = MultinomialNB(alpha = 2,fit_prior=False)
	
	#Train the model using X_train_dtm (timing it with an IPython "magic command")
	nb.fit(X_train_tf, y_train)

	#make class prediction for the same training data to calculate train accuracy
	y_train_class = nb.predict(X_train_tf)
	#calculate training accuracy
	np.mean(y_train_class == y_train)

	#make class predictions for X_test_dtm
	y_pred_class = nb.predict(X_test_tf)
	#calculate accuracy of class predictions
	metrics.accuracy_score(y_test, y_pred_class)

	#Making a pipeline of vectorizer and classifier with necessary tuning params
	from sklearn.pipeline import Pipeline
	text_clf = Pipeline([('vect', CountVectorizer(analyzer = 'word',tokenizer = nltk.word_tokenize, stop_words = 'english', max_df = 0.4, min_df = 3,ngram_range=(1,1))),
                     	('tfidf', TfidfTransformer(use_idf=False)),
                     	('clf', MultinomialNB(alpha = 0.5,fit_prior=False)),

                    	])
                    
	#Train the model after CountVectorizer and TdidfTransformer in a pipeline
	text_clf.fit(X_train,y_train)

	#Make class prediction for X_test in pipeline i.e. X_test is converted to document-term matrix first and then transformed.
	predicted = text_clf.predict(X_test)

	#finding the accuracy of the model
	np.mean(predicted == y_test)

	# import and instantiate a logistic regression model
	from sklearn.linear_model import LogisticRegression
	logreg = LogisticRegression(C=10)

	# train the model using X_train_dtm
	logreg.fit(X_train_tf, y_train)

	# import and instantiate a SGDClassifier
	from sklearn.linear_model import SGDClassifier
	sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=0.001,n_iter=5, random_state=42)

	# train the model using X_train_dtm
	sgd.fit(X_train_tf, y_train)
	
	# Bagging
	from sklearn.ensemble import BaggingClassifier
	from sklearn.tree import DecisionTreeClassifier
	c=DecisionTreeClassifier()
	numtrees=9
	bagging=BaggingClassifier(base_estimator=c,n_estimators=numtrees,random_state=7)

	bagging.fit(X_train_tf, y_train)

	#make class prediction for the same training data to calculate train accuracy
	y_train_class = bagging.predict(X_train_tf)
	#calculate training accuracy
	np.mean(y_train_class == y_train)

	#test accuracy
	y_pred_class = bagging.predict(X_test_tf)
	#calculate accuracy of class predictions
	metrics.accuracy_score(y_test, y_pred_class)


	new_bug = name
	new = [new_bug]
	X_new_counts = count_vect.transform(new)
	X_new_tfidf = tf_transformer.transform(X_new_counts)
	predicted = nb.predict(X_new_tfidf)
	predicted1 = logreg.predict(X_new_tfidf)
	predicted2 = sgd.predict(X_new_tfidf)
	predicted3 = bagging.predict(X_new_tfidf)
	print("The priority Level Assigned to", '"',new_bug,'"', " is: #")
	print("Bootstrap aggregating (Bagging): ", priority.get(predicted3[0]),"#")
	print("Multinomial Naive Bayes: ", priority.get(predicted[0]),"#")
	print("Logistic Regression: ", priority.get(predicted1[0]),"#")
	print("Stochastic Gradient Descent Classifier: ", priority.get(predicted2[0]),"#")
	priority_level = (predicted + predicted1 + predicted2 + predicted3)/4
	print("Priority level from different classifiers is -> [ P",int(priority_level),"]")
	
if __name__ == '__main__':
	arg_parser  = argparse.ArgumentParser()
	arg_parser.add_argument("input",
				help = "input for program")
	args = arg_parser.parse_args()
	get_output(args.input)