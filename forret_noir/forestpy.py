import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
import seaborn as __sns
import math as __mt
# Transformer
from sklearn.preprocessing import PowerTransformer
# Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
# Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Encoder
from sklearn.preprocessing import OneHotEncoder
# Split
from sklearn.model_selection import train_test_split
# Balance Data
from imblearn.over_sampling import SMOTE
# Error Matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
# misc
import pickle

# plotting continuos variables
def plot_continous(df):
    
    df = df.select_dtypes(__np.number)
    for item in df.columns:
        __sns.displot(x=item, data = df, kde=True)
    __plt.show()
    
    
# plotting discrete variables    
def plot_discrete(df):
    
    r = __mt.ceil(df.shape[1]/2)
    c = 2
    fig, ax = __plt.subplots(r,c, figsize=(15,40))
    i = 0
    j = 0
    for item in df.columns:
        __sns.histplot(x=item, data = df, ax = ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    __plt.show()
    
    
# plotting boxplot    
def boxplot_continous(df):
    
    r = __mt.ceil(df.shape[1]/2)
    c = 2
    fig, ax = __plt.subplots(r,c, figsize=(15,20))
    i = 0
    j = 0

    for item in df.columns:
        __sns.boxplot(x=item, data=df, ax=ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    __plt.show() 


# plotting correlation matrix
def corr_mat(df):
    
    num = df.select_dtypes(__np.number)
    corr_matrix_cont=num.corr(method='pearson')
    fig, ax = __plt.subplots(figsize=(10, 8))
    ax = __sns.heatmap(corr_matrix_cont, annot=True)
    __plt.show()
    
    
# transform variables and plot the transformed variables
def plot_transformer(df): 
    
    data_log = __pd.DataFrame()
    data_log = log_it(df)
    data_bc, data_yj = power_transform(df)
    r = df.shape[1]
    c = 4
    fig, ax = __plt.subplots(r, c, figsize=(30,30))
    i = 0
    data = ""
    for item in df.columns:
        for j in range(c):
            if j == 0:
                data = df
                head = "original"
            elif j == 1:
                data = data_log
                head = "log"
            elif j == 2:
                data = data_yj
                head = "yeo-johnson"
            elif j == 3:
                data = data_bc
                head = "box-cox"
            ax[0,j].set_title(head)
         
            if item in data.columns:
                __sns.distplot(a = data[item], ax = ax[i, j]) 
        i = i + 1
    __plt.show()
    
    
# perform log transform        
def log_it(df):
    
    data_log = __pd.DataFrame()
    for item in df.columns:
        data_log[item] = df[item].apply(__log_transform_clean)
        
    return data_log


def __log_transform_clean(x):
    
    if __np.isfinite(x) and x!=0:
        return __np.log(x)
    else:
        return __np.NAN
    
    
def __df_box_cox(df):
    df1 = __pd.DataFrame()
    for item in df.columns:
        if df[item].min() > 0:
            df1[item] = df[item]
            
    return df1

# perform power transform
def power_transform(df):
    
    df_f_bc = __df_box_cox(df)
    pt_bc = PowerTransformer(method="box-cox")
    pt_bc.fit(df_f_bc)
    df_bc = __pd.DataFrame(pt_bc.transform(df_f_bc), columns = df_f_bc.columns)
    
    pt_yj = PowerTransformer()
    pt_yj.fit(df)
    df_yj = __pd.DataFrame(pt_yj.transform(df), columns = df.columns)
    
    return df_bc, df_yj

# 
def display_filledna(df):
    fig, ax = __plt.subplots(1,3,figsize=(16,5))
    __sns.histplot(df, ax = ax[0])
    __sns.histplot(df.fillna(__np.mean(df)), ax = ax[1])
    __sns.histplot(df.fillna(__np.median(df[df.notna()])),ax = ax[2])
    __plt.show()


# remove outliers
def remove_outliers(df):
    rem_df = df.copy()
    df_num = rem_df.select_dtypes(__np.number)
    #df_cat = df.select_dtypes(__np.number)
    #df_other = df.select_dtypes(exclude=[__np.object, __np.number])

    old_rows = df.shape[0]
    for item in df_num.columns:
        print(item)
        iqr = __np.nanpercentile(df[item],75) - __np.nanpercentile(df[item],25)
        if iqr > 0:
            print(iqr)
            upper_limit = __np.nanpercentile(df[item],75) + 1.5*iqr
            print(upper_limit)
            lower_limit = __np.nanpercentile(df[item],25) - 1.5*iqr
            print(lower_limit)
            rem_df[item] = rem_df[(rem_df[item] < upper_limit) & (rem_df[item] > lower_limit)][item]
        
    rows_removed = old_rows - df.shape[0]
    rows_removed_percent = (rows_removed/old_rows)*100
        
    print("{} rows have been removed, {}% in total".format(rows_removed, rows_removed_percent))
      
    return rem_df


# checks for unique values
def unique(df):
    
    for item in df.columns:
        print(item)
        print(df[item].unique())
        print(df[item].value_counts(dropna=False))
        print("---------------")

        
# builds model and saves it        
def linear_regression(X_test, y_test, filename, X_train = None, y_train = None, train = True):
    
    if train:
        knn_models = __search_k(X_train, y_train, X_test, y_test)
        var = int(input("Please enter k:"))
        files = []
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        filename_lr = filename + "_linear.sav"
        pickle.dump(lr, open("models/"+filename_lr, 'wb'))
        print("-----------------------------")
        print("------Linear Regression------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",__np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("-----------------------------")
        print("------Linear Regression------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",__np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        print("Filename Linear: " + filename_lr)
        files.append(filename_lr)
        
        knn_models[var-2].score(X_test, y_test)
        y_pred_train = knn_models[var-2].predict(X_train)
        y_pred_test = knn_models[var-2].predict(X_test)
        filename_knn = filename + "_knn.sav"
        pickle.dump(knn_models[var-2], open("models/"+filename_knn, 'wb'))
        print("--------------KNN------------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",__np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("Filename knn: " + filename_knn)
        print("-----------------------------")
        print("--------------KNN------------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",__np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        files.append(filename_knn)
        
        return files
    
    if train == False:
        loaded_linear = pickle.load(open("models/"+filename[0], 'rb'))
        y_pred = loaded_linear.predict(X_test)
        print("------Linear Regression------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred))
        print("MSE:",mean_squared_error(y_test , y_pred))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred)))
        print("MAE:",mean_absolute_error(y_test , y_pred))
        print("-----------------------------")
        
        loaded_knn = pickle.load(open("models/"+filename[1], 'rb'))
        y_pred1 = loaded_knn.predict(X_test)
        print("--------------KNN------------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred1))
        print("MSE:",mean_squared_error(y_test , y_pred1))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred1)))
        print("MAE:",mean_absolute_error(y_test , y_pred1))
        print("-----------------------------")
        
        return y_pred, y_pred_1

def logistic_regression(X_test, y_test, filename, labels, X_train = None, y_train = None, train = True):
    
    if train:
        
        lr1 = LogisticRegression(random_state=0, solver='lbfgs')
        lr1.fit(X_train, y_train)
        y_pred_train = lr1.predict(X_train)
        y_pred_test = lr1.predict(X_test)
        filename_lr = filename + "_logistic.sav"
        pickle.dump(lr1, open("models/"+filename_lr, 'wb'))
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
        print("Filename Linear: " + filename_lr)
        cm_test = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=lr1.classes_)
        disp.plot()
        __plt.show()

        return filename_lr
    
    if train == False:
        loaded_linear = pickle.load(open("models/"+filename, 'rb'))
        y_pred_test = loaded_linear.predict(X_test)
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=loaded_linear.classes_)
        disp.plot()
        __plt.show()
        
        return y_pred_test

def smote(X_train, y_train):
    sm = SMOTE(random_state=100,k_neighbors=3)
    X_train_SMOTE,y_train_SMOTE = sm.fit_resample(X_train,y_train)
    
    return X_train_SMOTE, y_train_SMOTE
        
def __search_k(X_train, y_train, X_test, y_test):
    knn_models = []
    scores = []
    for k in range(2,15):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        knn_models.append(model)
        scores.append(model.score(X_test, y_test))
    __plt.figure(figsize=(10,6))
    __plt.plot(range(2,15),scores,color = 'blue', linestyle='dashed',
    marker='o', markerfacecolor='red', markersize=10)
    __plt.title('R2-scores vs. K Value')
    __plt.xticks(range(1,16))
    __plt.gca().invert_yaxis()
    __plt.xlabel('K')
    __plt.ylabel('Accuracy')
    __plt.show()
    return knn_models

def standard(X, filename, fit = True):

    if fit:
        scaler = MinMaxScaler()
        scaler.fit(X)
        filename = filename + ".sav"
        pickle.dump(scaler, open("scaler/"+filename, 'wb'))
        X_scaled = scaler.transform(X)
        X_scaled_df = __pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled_df, filename
    
    if fit == False:
        loaded_model = pickle.load(open("scaler/"+filename, 'rb'))
        X_scaled = loaded_model.transform(X)
        X_scaled_df = __pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled_df


def min_max(X, filename, fit = True):
  
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(X)
        filename = filename + ".sav"
        pickle.dump(scaler, open("scaler/"+filename, 'wb'))
        X_scaled = scaler.transform(X)
        X_scaled_df = __pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled_df, filename
    
    if fit == False:
        loaded_model = pickle.load(open("scaler/"+filename, 'rb'))
        X_scaled = loaded_model.transform(X)
        X_scaled_df = __pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled_df
    
def one_hot(X, filename, y=None, fit = True):
    
    X = X.copy()

    if fit:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
        X_num_train = X_train.select_dtypes(__np.number).reset_index(drop=True)
        X_cat_train = X_train.select_dtypes(__np.object)
        X_num_test = X_test.select_dtypes(__np.number).reset_index(drop=True)
        X_cat_test = X_test.select_dtypes(__np.object)
        X = X.select_dtypes(__np.object)
        
        col_list = [X[col].unique() for col in X.columns]
        encoder = OneHotEncoder(handle_unknown='error', drop='first', categories=col_list)
        
        encoder.fit(X_cat_train)
        filename = filename + ".sav"
        pickle.dump(encoder, open("encoder/"+filename, 'wb'))
        categoricals_encoded_train = encoder.transform(X_cat_train).toarray()
        categoricals_encoded_test = encoder.transform(X_cat_test).toarray()
        categoricals_encoded_train_df = __pd.DataFrame(categoricals_encoded_train, columns = encoder.get_feature_names_out())
        categoricals_encoded_test_df = __pd.DataFrame(categoricals_encoded_test, columns = encoder.get_feature_names_out())
        df_train_onehot = __pd.concat([categoricals_encoded_train_df, X_num_train], axis=1)
        df_test_onehot = __pd.concat([categoricals_encoded_test_df, X_num_test], axis=1)
        
        return df_train_onehot, df_test_onehot, y_train, y_test, filename
    
    if fit == False:
        loaded_encoder = pickle.load(open("encoder/"+filename, 'rb'))
        categoricals_encoded = loaded_encoder.transform(X).toarray()
        categoricals_encoded = __pd.DataFrame(categoricals_encoded, columns = loaded_encoder.get_feature_names_out())
        return categoricals_encoded

def power_e(x):
    return np.e**x

# Display NaN
def show_nan(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            print("The column", col, "has", df[col].isna().sum(), "NaN" )

# Display empty strings
def show_empty_variables(df):
    df = df.select_dtypes([__np.number, __np.object])
    columns = df.columns
    for col in columns:
        if len(df[df[col].isin(["", " "])][col].value_counts()) == 1:
            empty_amount = df[df[col].isin(["", " "])][col].value_counts()[0]
            percent_empty = __np.round((empty_amount / df.shape[0])*100, 2)
            print("The column", col, "has", empty_amount, "empty strings which means", percent_empty, "%" )
            
# remove columns with certain NaN
def remove_cols_with_na(df, thr = 0.25):
    nulls_percent_df = __pd.DataFrame(df.isna().sum()/len(data)).reset_index()
    nulls_percent_df.columns = ['column_name', 'nulls_percentage']
    return list(nulls_percent_df[nulls_percent_df['nulls_percentage'] > thr]['column_name'].values)