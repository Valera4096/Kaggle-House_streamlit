import joblib

import streamlit as st
import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")

from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.cat_boost import CatBoostEncoder
from catboost import CatBoostRegressor



class MasVnrType_modify(BaseEstimator, TransformerMixin): 
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_copy = X.copy()

        ff = ((X_copy['MasVnrArea'] == 0)  & (X_copy['MasVnrType'] != 'NO'))
        X_copy.loc[ff,'MasVnrArea'] = X_copy['MasVnrArea'].loc[X_copy['MasVnrType'] == 'Stone'].median()
        
        f = ((X_copy['MasVnrArea'] != 0)  & (X_copy['MasVnrType'] == 'NO'))
        X_copy.loc[f,'MasVnrType'] = 'BrkFace'
        
        return X_copy
    
 
data_df = pd.read_csv('train.csv')
original_date = pd.read_csv('file.csv')


ml_pipeline = joblib.load('model_catboost.pkl')

st.title('Предсказанние цены на дом')


st.header('Заполните данные')

GrLivArea = st.number_input('Above grade (ground) living area square feet / Жилая площадь над уровнем земли в квадратных футах')

GarageType = st.selectbox(
    'Garage location / расположение гаража ?',
    ('Attchd', 'Detchd', 'BuiltIn', 'CarPort','Basment', '2Types'))

ExterCond = st.selectbox(
    'Present condition of the material on the exterior / состояние материала снаружи ?',
    ('TA', 'Gd', 'Fa', 'Po', 'Ex'))

Fireplaces = st.slider('Number of fireplaces / Кол - во каминов ?  ',1,10)

EnclosedPorch = st.slider('Enclosed porch area in square feet / Площадь  веранды?  ',1,400)

TotRmsAbvGrd = st.slider('Total rooms above grade (does not include bathrooms) / Общее кол- во комнат не включая санузла ?  ',1,30)


entered_dataframe = pd.DataFrame({'GrLivArea':[GrLivArea],
                                  'GarageType':[GarageType],
                                  'ExterCond':[ExterCond],
                                  'Fireplaces':[Fireplaces],
                                  'EnclosedPorch':[EnclosedPorch],
                                  'TotRmsAbvGrd':[TotRmsAbvGrd]
                                  })

original_date.drop(entered_dataframe.columns.to_list(), axis= 1, inplace= True)


res_df = pd.concat([entered_dataframe,original_date],axis=1)
res_df.drop('SalePrice',axis=1 , inplace= True)

res = ml_pipeline.predict(res_df)


if st.button('Расчитать цену'):
    st.title('Предсказанная цена')
    st.header("Результат: " + str(round(*np.exp(res), 2)) + " $")
    








