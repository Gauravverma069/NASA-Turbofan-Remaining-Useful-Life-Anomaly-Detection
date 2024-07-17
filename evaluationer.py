# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error,r2_score,mean_squared_error,root_mean_squared_log_error,mean_absolute_error,mean_squared_log_error
from sklearn.metrics import f1_score, accuracy_score, precision_score,recall_score, average_precision_score
# creating a class for evaluation
    
reg_evaluation_df = pd.DataFrame({"evaluation_df_method" :[],
                                "model": [],# model displays regression model
                                "method": [],# method display evaluation metrics used
                                "train_r2": [],# train r2 shows train R2 score
                                "test_r2": [],# test r2 shows test R2 Score
                                "adjusted_r2_train": [],# adjusted_r2_train shows adjusted r2 score for train
                                "adjusted_r2_test": [],# adjusted_r2_test shows adjusted r2 score for test
                                "train_evaluation": [],# train_evaluation shows train evaluation score by used method
                                "test_evaluation" : []# test_evaluation shows test evaluation score by used method
                            })

classification_evaluation_df = pd.DataFrame({"evaluation_df_method" :[],
                        'model': [],
                        'train_f1': [],
                        'test_f1': [],
                        'train_acc': [],
                        'test_acc': [],
                        'precision_train': [],
                        'precision_test': [],
                        'recall_train': [],
                        'recall_test': []
                    })

# function for evaluating dataframe
def evaluation(evaluation_df_method,X_train,X_test,y_train,y_test,model,method,eva):# input parameters from train_test_split , model and method for evaluation.
    global y_pred_train,y_pred_test,y_pred_proba_train,y_pred_proba_test
    model = model
    model.fit(X_train,y_train) # model fitting
    y_pred_train = model.predict(X_train) # model prediction for train
    y_pred_test = model.predict(X_test) # model prediction for test

    if eva == "reg":
        
        train_r2 = r2_score(y_train, y_pred_train) # evaluating r2 score for train
        test_r2 = r2_score(y_test, y_pred_test)  # evaluating r2 score for test
        
        n_r_train, n_c_train = X_train.shape # getting no of rows and columns of train data
        n_r_test,  n_c_test = X_test.shape # getting no of rows and columns of test data
        
        adj_r2_train = 1 - ((1 - train_r2)*(n_r_train - 1)/ (n_r_train - n_c_train - 1))  # evaluating adjusted r2 score for train
        adj_r2_test = 1 - ((1 - test_r2)*(n_r_test - 1)/ (n_r_test - n_c_test - 1)) # evaluating adjusted r2 score for test
    
        train_evaluation = method(y_train, y_pred_train) # evaluating train error
        test_evaluation = method(y_test, y_pred_test) # evaluating test error
        
        if method == root_mean_squared_error:
            a = "root_mean_squared_error"
        elif method ==root_mean_squared_log_error:
            a = "root_mean_squared_log_error"
        elif method == mean_absolute_error:
            a = "mean_absolute_error"
        elif method == mean_squared_error:
            a = "mean_squared_error"
        elif method == mean_squared_log_error:
            a = "mean_squared_log_error"    
        
        # declaring global dataframes
        global reg_evaluation_df,temp_df
        
        # creating temporary dataframe for concating in later into main evaluation dataframe
        temp_df = pd.DataFrame({"evaluation_df_method" :[evaluation_df_method],
                                    "model": [model],
                                    "method": [a],
                                    "train_r2": [train_r2],
                                    "test_r2": [test_r2],
                                    "adjusted_r2_train": [adj_r2_train],
                                    "adjusted_r2_test": [adj_r2_test],
                                    "train_evaluation": [train_evaluation],
                                    "test_evaluation" : [test_evaluation]
                                    })
        reg_evaluation_df = pd.concat([reg_evaluation_df,temp_df]).reset_index(drop = True)
        


        
        return reg_evaluation_df # returning evaluation_df

    elif eva == "class":
                
        # y_pred_proba_train= model.predict_proba(X_train)
        # y_pred_proba_test= model.predict_proba(X_test)

        unique_classes = np.unique(y_train)
    
        # Determine the average method
        if len(unique_classes) == 2:
            # Binary classification
            print("Using 'binary' average for binary classification.")
            average_method = 'binary'
        elif len(unique_classes)!=2:
            # Determine the distribution of the target column
            class_counts = np.bincount(y_train)
            
            # Check if the dataset is imbalanced
            imbalance_ratio = max(class_counts) / min(class_counts)
            
            if imbalance_ratio > 1.5:
                # Imbalanced dataset
                print("Using 'weighted' average due to imbalanced dataset.")
                average_method = 'weighted'
            else:
                # Balanced dataset
                print("Using 'macro' average due to balanced dataset.")
                average_method = 'macro'
            
        # F1 scores
        train_f1_scores = (f1_score(y_train, y_pred_train,average=average_method))
        test_f1_scores = (f1_score(y_test, y_pred_test,average=average_method))    
    
        # Accuracies
        train_accuracies = (accuracy_score(y_train, y_pred_train))
        test_accuracies = (accuracy_score(y_test, y_pred_test))
    
        # Precisions
        train_precisions = (precision_score(y_train, y_pred_train,average=average_method))
        test_precisions = (precision_score(y_test, y_pred_test,average=average_method))
    
        # Recalls
        train_recalls = (recall_score(y_train, y_pred_train,average=average_method))
        test_recalls = (recall_score(y_test, y_pred_test,average=average_method))
        
        # declaring global dataframes
        global classification_evaluation_df,temp_df1
        
        # creating temporary dataframe for concating in later into main evaluation dataframe
        temp_df1 = pd.DataFrame({"evaluation_df_method" :[evaluation_df_method],
            'model': [model],
            'train_f1': [train_f1_scores],
            'test_f1': [test_f1_scores],
            'train_acc': [train_accuracies],
            'test_acc': [test_accuracies],
            'precision_train': [train_precisions],
            'precision_test': [test_precisions],
            'recall_train': [train_recalls],
            'recall_test': [test_recalls]
        })
        classification_evaluation_df = pd.concat([classification_evaluation_df, temp_df1]).reset_index(drop = True)
        
        return classification_evaluation_df # returning evaluation_df

global method_df
method_df = pd.DataFrame(data = [root_mean_squared_error, root_mean_squared_log_error,mean_absolute_error,mean_squared_error,mean_squared_log_error],
                         index = ["root_mean_squared_error", "root_mean_squared_log_error","mean_absolute_error","mean_squared_error","mean_squared_log_error"])