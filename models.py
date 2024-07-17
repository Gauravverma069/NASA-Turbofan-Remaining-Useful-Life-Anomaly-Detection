import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import algorithms for classification 
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier,XGBRFClassifier 
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB,CategoricalNB
# import algorithms for regression
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB



# dictionary where keys are name of algorithm and values are algorithm for classifier
algos_class = {
    "Logistic Regression": LogisticRegression(),
    "SGD Classifier": SGDClassifier(),
    "Ridge Classifier": RidgeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Hist Gradient Boosting Classifier": HistGradientBoostingClassifier(),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "SVC": SVC(),
    "XGB Classifier": XGBClassifier(),
    "XGBRF Classifier": XGBRFClassifier(),
    "MLP Classifier": MLPClassifier(),
    "LGBM Classifier": LGBMClassifier(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Categorical Naive Bayes": CategoricalNB()}

# dictionary where keys are name of algorithm and values are algorithm for regression
algos_reg = {
    "Linear Regression": LinearRegression(),
    "Ridge Regressor": Ridge(),
    "Lasso Regressor": Lasso(),
    "ElasticNet Regressor": ElasticNet(),
    "Random Forest Regressor": RandomForestRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(),
    "K Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "XGB Regressor": XGBRegressor(),
    "XGBRF Regressor": XGBRFRegressor(),
    "MLP Regressor": MLPRegressor(),
    "LGBM Regressor": LGBMRegressor(),
    "Gaussian Naive Bayes": GaussianNB()}

# dataframe where index are name of algorithm as "algorithm name" , column  is algorithm as "algorithm"

Classification_models = pd.DataFrame(data=algos_class.values(), index=algos_class.keys())

Regression_models = pd.DataFrame(data=algos_reg.values(), index=algos_reg.keys())

