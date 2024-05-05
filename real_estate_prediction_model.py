from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
from bayes_opt import BayesianOptimization
import os
import pandas as pd
import numpy as np

use_columns = ['市区町村名','地区名','最寄駅：名称','最寄駅：距離（分）','取引価格（総額）','延床面積（㎡）','建築年','建物の構造','用途','前面道路：種類','前面道路：幅員（ｍ）','取引時期']
# ログ設定
import logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

Front_Road_Type_mapping = {
    '市': '公',
    '区': '公',
    '町': '公',
    '村': '公',
    '都': '公',
    '道': '公',
    '府': '公',
    '県': '公',
    '区画街路': 'その他',
    '林道': 'その他',
    '農道': 'その他',
    '道路': 'その他'
}

Building_Structure_mapping={
    'ブロック造':'木造',
    'ＲＣ、木造':'ＲＣ',
    'ＲＣ、軽量鉄骨造':'ＲＣ',
    '鉄骨造、木造':'木造',
    'ＲＣ、ブロック造':'ＲＣ',
    'ＲＣ、鉄骨造':'ＲＣ',
    '木造、ブロック造':'木造',
    '木造、軽量鉄骨造':'木造',
    '鉄骨造、ブロック造':'鉄骨造',
    'ＲＣ、木造、ブロック造':'木造',
    'ＳＲＣ、鉄骨造':'鉄骨造',
    'ＳＲＣ、ＲＣ':'ＲＣ'
}

def preprocess_data(data):
    print(data)
    data['建築年'] = data['建築年'].str.replace('年', '').str.replace('戦前', '1945')
    data['建築年'] = pd.to_numeric(data['建築年'], errors='coerce').fillna(data['建築年'].median()).astype(int)
    if data['延床面積（㎡）'].dtype != 'int64':
        data['延床面積（㎡）'] = data['延床面積（㎡）'].replace('2,000㎡以上', '2000').str.replace(',', '')        
    data['延床面積（㎡）'] = data['延床面積（㎡）'].astype(float).fillna(data['延床面積（㎡）'].median()).astype(int)
    data['前面道路：種類'] = data['前面道路：種類'].replace(Front_Road_Type_mapping, regex=True)  
    def convert_quarter_to_last_month(date_str):
        year, quarter = date_str.split('年第')
        quarter = int(quarter[0])
        month = 3 * quarter
        return f"{year}-{month:02d}-01"
    data['取引時期'] = data['取引時期'].apply(convert_quarter_to_last_month)
    data['取引時期'] = pd.to_datetime(data['取引時期'])    
    
    bins = [0, 4, 6, 8, float('inf')]
    labels = ['4m未満', '4m以上6m未満', '6m以上8m未満', '8m以上']
    data['前面道路：幅員（ｍ）'] = pd.cut(data['前面道路：幅員（ｍ）'], bins=bins, labels=labels, right=False)
    #取引価格に対して対数変換を適用
    data['対数変換後_取引価格（総額）'] = np.log1p(data['取引価格（総額）'])
    data['築年数']=data['取引時期'].dt.year - data['建築年']
    data['新築フラグ']=data['築年数']=0
    
    return data

def xgb_evaluate(max_depth, learning_rate, n_estimators, model, X_train, y_train):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    model.set_params(regressor__max_depth=max_depth, regressor__learning_rate=learning_rate, regressor__n_estimators=n_estimators)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=3).mean()

def build_and_train_model(X_train, y_train):
    categorical_features = ['市区町村名', '地区名', '最寄駅：名称', '建物の構造', '用途', '前面道路：種類', '前面道路：幅員（ｍ）']
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(device='gpu'))  
    ])



    # xgb_evaluate function needs to be defined to use in Bayesian Optimization.
    def xgb_evaluate(max_depth, learning_rate, n_estimators):
        params = {
            'regressor__max_depth': int(max_depth),
            'regressor__learning_rate': learning_rate,
            'regressor__n_estimators': int(n_estimators)
        }
        model.set_params(**params)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)  # Assuming score is to be maximized
        return score

    optimizer = BayesianOptimization(
        f=lambda max_depth, learning_rate, n_estimators: xgb_evaluate(max_depth, learning_rate, n_estimators),
        pbounds={'max_depth': (3, 10), 'learning_rate': (0.01, 0.3), 'n_estimators': (50, 300)},
        random_state=1
    )

    optimizer.maximize(init_points=5, n_iter=25)

    best_params = optimizer.max['params']
    model.set_params(regressor__max_depth=int(best_params['max_depth']),
                     regressor__n_estimators=int(best_params['n_estimators']),
                     regressor__learning_rate=best_params['learning_rate'])
    
    model.fit(X_train, y_train)
    joblib.dump(model, 'best_model.pkl')
    print("Best model saved with Bayesian Optimization.")
    
    return model


def postprocess_predictions(predictions):
    # 対数変換された予測値を元に戻す
    predictions = np.expm1(predictions)
    # 念のため、負の値を0に設定
    predictions = np.maximum(predictions, 0)
    return predictions

def main():
    try:
        data = pd.read_csv(os.getcwd() + '/input/' +  'Tokyo_20201_20234.csv', encoding='cp932')        
        data = data[use_columns]
        data = data[data['用途'].isin(['住宅', '共同住宅'])]
        data = preprocess_data(data)
        
        # 外れ値の除去（3シグマ法）
        price_mean = data['取引価格（総額）'].mean()
        price_std = data['取引価格（総額）'].std()
        lower_bound = price_mean - 3 * price_std
        upper_bound = price_mean + 3 * price_std
        data = data[(data['取引価格（総額）'] >= lower_bound) & (data['取引価格（総額）'] <= upper_bound)]        
        
        X = data.drop(['取引価格（総額）','対数変換後_取引価格（総額）'], axis=1)
        y = data['対数変換後_取引価格（総額）']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = build_and_train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'平均二乗誤差: {mse}')
        
        new_data = pd.read_csv(os.getcwd() + '/input/' + 'new_data.csv', encoding='cp932')
        #new_data = pd.read_csv(os.getcwd() + '/input/' + 'for_API_test.csv', encoding='cp932')
        new_data=new_data[use_columns]
        new_data = preprocess_data(new_data)
        loaded_model = joblib.load('best_model.pkl')
        raw_predictions = loaded_model.predict(new_data) 
        #対数から元に戻す      
        final_predictions = postprocess_predictions(raw_predictions)  # 逆変換とポストプロセッシング
        
        df = pd.DataFrame({'用途': new_data['用途'],'正解値': new_data['取引価格（総額）'], '予測値': pd.Series(final_predictions).astype('int64')})
        df.to_csv(os.getcwd() + '/output/' + 'test.csv')
        print(df)
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

    #print('end')
def prediction_service(df):
    df = preprocess_data(df)
    loaded_model = joblib.load('best_model.pkl')
    raw_predictions = loaded_model.predict(df) 
    #対数から元に戻す      
    final_predictions = postprocess_predictions(raw_predictions)  # 逆変換とポストプロセッシング
    return final_predictions

if __name__ == '__main__':
    main()