#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from Methods import select_k_best_features, best_parameter_from_search, precision_n_recall
from calculation import computeFraction
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import cross_validation

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Get the total number of pois in the dataset
num_poi = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        print data_dict[name]['email_address']
        print data_dict[name]
        num_poi += 1
num_tut = 0
for name in data_dict.keys():
    num_tut += 1
print "There are", num_poi, "POIs in total."
print "There are", num_tut, "Users data in total."
num_in = num_tut - num_poi
print "There are", num_in, "Non POIs data in total."
### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Create new features
for name in my_dataset:
    my_point = my_dataset[name]
    ### fraction of emails one person got from poi
    my_point["fraction_from_poi"] = \
    computeFraction( my_point["from_poi_to_this_person"], my_point["to_messages"] )
    ### fraction of emails one person sent to poi
    my_point["fraction_to_poi"] = \
    computeFraction( my_point["from_this_person_to_poi"], my_point["from_messages"] )
    ### number of poi as shared recipients per received email 
    my_point["shared_poi_per_email"] = \
    computeFraction( my_point["shared_receipt_with_poi"], my_point["to_messages"] )
    ### indictor of whether a person has an email address
    if my_point['email_address'] == 'NaN':
        my_point['email_exists'] = 0
    else:
        my_point['email_exists'] = 1



### Full list of features (always starts with 'poi' -- the value to predict), 
### including the both original and newly created ones, expect for 'email_adress', 
### which is more of an id rather than a feature.
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', \
                 'loan_advances', 'bonus', 'restricted_stock_deferred', \
                 'deferred_income', 'total_stock_value', 'expenses', \
                 'exercised_stock_options', 'other', 'long_term_incentive', \
                 'restricted_stock', 'director_fees', 'to_messages', \
                 'email_exists', 'fraction_from_poi', 'fraction_to_poi', \
                 'shared_poi_per_email','from_messages',
                 'shared_receipt_with_poi']
my_feature_list = features_list+['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                                 'shared_receipt_with_poi', 'fraction_to_poi']

target_label = 'poi'
num_features = 10 

# function using SelectKBest to find best features
def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    print scores
    return k_best_features


best_features = get_k_best(my_dataset, my_feature_list, num_features)

my_feature_list = [target_label] + best_features.keys()

# 3.6 print features
print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

data = featureFormat(my_dataset, my_feature_list)
# split into labels and features
labels, features = targetFeatureSplit(data)

def callNBC():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)
    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
    ])
    tester_prep(clf, my_dataset, my_features_list)


def callKM():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep) 
    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', KMeans(n_clusters=2, tol=0.001))
    ])
    tester_prep(clf, my_dataset, my_features_list)

def callADA():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)
    #from sklearn import tree
    dt = DecisionTreeClassifier() 
    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1))
    ])
    tester_prep(clf, my_dataset, my_features_list)

def callGRA():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)
    #dt = DecisionTreeClassifier() 
    

    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1))
    ])


    tester_prep(clf, my_dataset, my_features_list)

### Random Forest Classifier with best K features
def callRFC():
    ### Feature selection
    num_to_keep = 9
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)

    
    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(max_depth = 5, 
                                 max_features = 'sqrt', 
                                 n_estimators = 10, 
                                 random_state = 42))

    ])

    tester_prep(clf, my_dataset, my_features_list)



def callLR():
    ### Feature selection
    num_to_keep = 16
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)

   ##
    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', 
                                              random_state = 42))
    ])

    tester_prep(clf, my_dataset, my_features_list)


def callSVC():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)

  
    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel = 'rbf', C = 1000, gamma = 0.0001, 
                               random_state = 42, class_weight = 'auto'))
    ])

    tester_prep(clf, my_dataset, my_features_list)


def tester_prep(clf, my_dataset, my_features_list):
    print("Testing the performance of the classifier...")
    test_classifier(clf, my_dataset, my_features_list)

    ### Dump classifier, dataset, and features_list so anyone can run/check the results.
    dump_classifier_and_data(clf, my_dataset, my_features_list)
### 3.6 Evaluate all functions

print "Select Algorithm: \n"
print "1. Naive Bayes\n"
print "2. Logistic Regression\n"
print "3. SVC\n"
print "4. Random Forest\n";
print "5. K means\n";
print "6. Ada Boost\n";
print "7. Gradient Boost\n";
print "8. Run All the Algorithms\n";
#print "Select an Algorithm"
a = input('Select an Algorithm:')

if (a == 1):
    print "Running Naive Bayes Classifier\n"
    callNBC()

if (a == 2):
    print "Running Logistic Regression Classifier\n"
    callLR()

if ( a == 3):
    print "Running SVC Classifier\n"
    callSVC()
    
if ( a == 4):
    print "Running Random Forest Classifier\n"
    callRFC()
    
if ( a == 5):
    print "Running K Means Classifier\n"
    callKM()

if ( a == 6):
    print "Running Ada Boost Classifier\n"
    callADA()

if ( a == 7):
    print "Running Gradient Boost Classifier\n"
    callGRA()

if ( a == 8):
    print "Running SVC Classifier\n"
    callSVC()
    print "Running Random Forest Classifier\n"
    callRFC()
    print "Running Naive Bayes Classifier\n"
    callNBC()
    print "Running Logistic Regression Classifier\n"
    callLR()
    print "Running Kmeans Classifier\n"
    callKM()
    print "Running ADA Boost Classifier\n"
    callADA()
    print "Running Gradient Boost Classifier\n"
    callGRA()



