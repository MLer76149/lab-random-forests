from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
import seaborn as __sns
import math as __mt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

def decision_tree(X_test, y_test, filename, random_grid, X_train = None, y_train = None, train = True):
    
    if train:
        regr = DecisionTreeRegressor(max_depth=random_grid['max_depth'],
                             criterion = random_grid['criterion'],
                             min_samples_split=random_grid['min_samples_split'],
                             min_samples_leaf = random_grid['min_samples_leaf'],
                             max_features = random_grid['max_features'])
        model = regr.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        filename_lr = filename + "_decision_tree.sav"
        pickle.dump(model, open("models/"+filename_lr, 'wb'))
        print("-----------------------------")
        print("--------Decision Tree--------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",__np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("-----------------------------")
        print("--------Decision Tree--------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",__np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        print("Filename Linear: " + filename_lr)

        return filename_lr
    
    if train == False:
        loaded_linear = pickle.load(open("models/"+filename[0], 'rb'))
        y_pred = loaded_linear.predict(X_test)
        print("--------Decision Tree--------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred))
        print("MSE:",mean_squared_error(y_test , y_pred))
        print("RMSE:",__np.sqrt(mean_squared_error(y_test , y_pred)))
        print("MAE:",mean_absolute_error(y_test , y_pred))
        print("-----------------------------")

        return y_pred
    
def decision_tree_cross(X_test, y_test, filename, random_grid, X_train = None, y_train = None, train = True):
    
    if train:
        regr1 = DecisionTreeRegressor(max_depth=random_grid['max_depth'],
                             criterion = random_grid['criterion'],
                             min_samples_split=random_grid['min_samples_split'],
                             min_samples_leaf = random_grid['min_samples_leaf'],
                             max_features = random_grid['max_features'])
        results = cross_validate(regr1, X_train, y_train, cv = 5)
        print("The average R2 over the folds is: {:.2f}".format(results['test_score'].mean()))
        model = regr1.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        filename_lr = filename + "_decision_tree.sav"
        pickle.dump(model, open("models/"+filename_lr, 'wb'))
        print("-----------------------------")
        print("--------Decision Tree--------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",__np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("-----------------------------")
        print("--------Decision Tree--------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",__np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        print("Filename Linear: " + filename_lr)

        return filename_lr
    
    if train == False:
        loaded_linear = pickle.load(open("models/"+filename[0], 'rb'))
        y_pred = loaded_linear.predict(X_test)
        print("--------Decision Tree--------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred))
        print("MSE:",mean_squared_error(y_test , y_pred))
        print("RMSE:",__np.sqrt(mean_squared_error(y_test , y_pred)))
        print("MAE:",mean_absolute_error(y_test , y_pred))
        print("-----------------------------")

        return y_pred
    
def grid_search(X_train, y_train, grid, cv = 5, model = "regression"):
    
    if model == "regression":
        model = DecisionTreeRegressor()
        grid_search = GridSearchCV(estimator = model, param_grid = grid, cv = cv)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        print("The best R2 for the best hyperparameters is {:.2f}".format(grid_search.best_score_))
        
        return grid_search.best_params_
        
    if model == "classification":
        model = DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator = model, param_grid = grid, cv = cv)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        print("The best acurracy for the best hyperparameters is {:.2f}".format(grid_search.best_score_))
        
        return grid_search.best_params_
    
    if model == "forest_class":
        model = RandomForestClassifier()
        grid_search = GridSearchCV(estimator = model, param_grid = grid, cv = cv)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        print("The best acurracy for the best hyperparameters is {:.2f}".format(grid_search.best_score_))
        
        return grid_search.best_params_

def random_search(X_train, y_train, random_grid, cv = 5, n_iter = 25, model = "regression"):
    if model == "regression":
        model = DecisionTreeRegressor()
        random_search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter=n_iter, cv = cv, n_jobs = 2)
        random_search.fit(X_train,y_train)
        print(random_search.best_params_)
        print("The best R2 for the best hyperparameters is {:.2f}".format(random_search.best_score_))
        
        return random_search.best_params_
                  
    if model == "classification":
        model = DecisionTreeClassifier()
        random_search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter=n_iter, cv = cv, n_jobs = 2)
        random_search.fit(X_train,y_train)
        print(random_search.best_params_)
        print("The best accuracy for the best hyperparameters is {:.2f}".format(random_search.best_score_))
        
        return random_search.best_params_
    
    if model == "forest_class":
        model = RandomForestClassifier()
        random_search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter=n_iter, cv = cv, n_jobs = 2)
        random_search.fit(X_train,y_train)
        print(random_search.best_params_)
        print("The best accuracy for the best hyperparameters is {:.2f}".format(random_search.best_score_))
        
        return random_search.best_params_
    
    
def random_forest_class(X_test, y_test, forest_grid, filename, labels, X_train = None, y_train = None, train = True):
    
    if train:
    
        clf = RandomForestClassifier(max_depth=forest_grid['max_depth'],
                                 min_samples_split=forest_grid['min_samples_split'],
                                 min_samples_leaf =forest_grid['min_samples_leaf'],
                                 max_samples=forest_grid['max_samples'],
                                 random_state = forest_grid['random_state'])

        clf.fit(X_train, y_train)

        print("The accuracy for the Random Forest in the TRAIN set is {:.2f}".format(clf.score(X_train, y_train)))
        print("The accuracy for the Random Forest in the TEST  set is {:.2f}".format(clf.score(X_test, y_test)))

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        filename_fr = filename + "_forest.sav"
        pickle.dump(clf, open("models/"+filename_fr, 'wb'))
        print("-----------------------------")
        print("-----Logistic Regression-----")
        print("----------Train Set----------")
        print("-----------------------------")
        print("The accuracy in the TRAIN set is: {:.2f}".format(accuracy_score(y_train,y_pred_train)))
        print("The precision in the TRAIN set is: {:.2f}".format(precision_score(y_train,y_pred_train, pos_label=labels[1])))
        print("The recall in the TRAIN set is: {:.2f}".format(recall_score(y_train,y_pred_train, pos_label=labels[1])))
        print("The F1 in the TRAIN set is: {:.2f}".format(f1_score(y_train,y_pred_train, pos_label=labels[1])))
        print("The Cohen-Kappa-Score in the TRAIN set is: {:.2f}".format(cohen_kappa_score(y_train, y_pred_train, weights="quadratic", labels=labels)))
        print("-----------------------------")
        print("-----Logistic Regression-----")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("The accuracy in the TEST set is: {:.2f}".format(accuracy_score(y_test,y_pred_test)))
        print("The precision in the TEST set is: {:.2f}".format(precision_score(y_test,y_pred_test, pos_label=labels[1])))
        print("The recall in the TEST set is: {:.2f}".format(recall_score(y_test,y_pred_test, pos_label=labels[1])))
        print("The F1 in the TEST set is: {:.2f}".format(f1_score(y_test,y_pred_test, pos_label=labels[1])))
        print("The Cohen-Kappa-Score in the TEST set is: {:.2f}".format(cohen_kappa_score(y_test, y_pred_test, weights="quadratic", labels=labels)))
        print("-----------------------------")
        print("Filename: " + filename_fr)
        cm_test = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=clf.classes_)
        disp.plot()
        __plt.show()

        return filename_fr
    
    if train == False:
        loaded_forest = pickle.load(open("models/"+filename, 'rb'))
        y_pred_test = loaded_forest.predict(X_test)
        print("-----------------------------")
        print("-----Logistic Regression-----")
        print("-----------------------------")
        print("The accuracy in the TEST set is: {:.2f}".format(accuracy_score(y_test,y_pred_test)))
        print("The precision in the TEST set is: {:.2f}".format(precision_score(y_test,y_pred_test, pos_label=labels[1])))
        print("The recall in the TEST set is: {:.2f}".format(recall_score(y_test,y_pred_test, pos_label=labels[1])))
        print("The F1 in the TEST set is: {:.2f}".format(f1_score(y_test,y_pred_test, pos_label=labels[1])))
        print("The Cohen-Kappa-Score in the TEST set is: {:.2f}".format(cohen_kappa_score(y_test, y_pred_test, weights="quadratic", labels=labels)))
        print("-----------------------------")
        
        cm_test = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=loaded_forest.classes_)
        disp.plot()
        __plt.show()
        
        return y_pred_test

        
    