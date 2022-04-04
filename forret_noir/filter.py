from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
import pandas as __pd

def reduce_features_variance(df, threshold):
    scaler = MinMaxScaler()
    scaler.fit(df)
    numerical_scaled = scaler.transform(df)
    numerical_scaled = __pd.DataFrame(numerical_scaled, columns=df.columns)
    sel = VarianceThreshold(threshold=(threshold))
    sel = sel.fit(numerical_scaled)
    temp = sel.transform(numerical_scaled)
    temp_df = __pd.DataFrame(temp)
    temp_df.columns = sel.get_feature_names_out()
    
    return temp_df

def reduce_features_k_best(X, y, k):
    scaler = MinMaxScaler()
    scaler.fit(X)
    numerical_scaled = scaler.transform(X)
    numerical_scaled = __pd.DataFrame(numerical_scaled, columns=X.columns)
    kbest = SelectKBest(chi2, k=k)
    kbest.fit(numerical_scaled,y)
    X_new = kbest.transform(numerical_scaled) 
    selected_columns = [numerical_scaled.columns[index] for index, value in enumerate(kbest.get_support().tolist()) if value == True]
    selected = __pd.DataFrame(X_new, columns = selected_columns)
    
    return selected

def reduce_features_recursive(X, y,  model = "linear", features=20):
    scaler = MinMaxScaler()
    scaler.fit(X)
    numerical_scaled = scaler.transform(X)
    numerical_scaled = __pd.DataFrame(numerical_scaled, columns=X.columns)
    if model == "linear":
        lm = LinearRegression()
        rfe = RFE(lm, n_features_to_select=features, verbose=False)
        rfe.fit(numerical_scaled, y)
        df = __pd.DataFrame(data = rfe.ranking_, columns=['Rank'])
        df['Column_name'] = __pd.DataFrame(numerical_scaled).columns
        return df
    
   
    