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
on = st.toggle('Дополнительные параметры')

if on:
    LotArea = st.number_input('Lot size in square feet / Размер участка в квадратных футах')
    res_df['LotArea'] = LotArea
    
    MSSubClass = st.slider('The building class / Общий класс здания  ',1,100)
    res_df['MSSubClass'] = MSSubClass
    
    Street = st.selectbox(
    'Type of road access / Тип подъезда к дороге ?',
    ('Pave', 'Grvl'))
    res_df['Street'] = Street
    
    Alley = st.selectbox(
    'Type of alley access / Тип доступа к переулку ?',
    ('Pave', 'Grvl'))
    res_df['Alley'] = Alley
    
    LotShape = st.selectbox(
    'General shape of property / Общий вид недвижемости ?',
    ('Reg', 'IR1', 'IR2', 'IR3'))
    res_df['LotShape'] = LotShape
    
    LandContour = st.selectbox(
    'Flatness of the property / Уклон участка ?',
    ('Lvl', 'Bnk', 'Low', 'HLS'))
    res_df['LandContour'] = LandContour
    
    Utilities = st.selectbox(
    'Type of utilities available / Тип доступных утилит ?',
    ('AllPub', 'NoSeWa'))
    res_df['Utilities'] = Utilities
    
    LandSlope = st.selectbox(
    'Slope of property/ Уклон объекта ?',
    ('Gtl', 'Mod', 'Sev'))
    res_df['LandSlope'] = LandSlope
    
    Neighborhood = st.selectbox(
    'Physical locations within Ames city limits / Физическое местоположение в пределах города ?',
    ('CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
       'Blueste'))
    res_df['Neighborhood'] = Neighborhood
    
    Condition1 = st.selectbox(
    'Proximity to main road or railroad / Близость к главной или ж/д дороге ?',
    ('Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA',
       'RRNe'))
    res_df['Condition1'] = Condition1
    
    BldgType =st.selectbox(
    'Type of dwelling / Тип жилья ?',
    ('1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'))
    res_df['BldgType'] = BldgType
    
    OverallQual = st.slider('The building class / Общий оценка состояния ремонта',1,10)
    res_df['OverallQual'] = OverallQual
    
    YearBuilt = st.number_input('Original construction date / Год постройки')
    res_df['YearBuilt'] = YearBuilt
    
    YearRemodAdd = st.number_input('Remodel date / Год ремонта')
    res_df['YearRemodAdd'] = YearRemodAdd
    
    RoofMatl = st.selectbox(
    'Roof material / Материал крыши ?',
    ('CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv',
       'Roll', 'ClyTile'))
    res_df['RoofMatl'] = RoofMatl
    
    Exterior1st = st.selectbox(
    'Exterior covering on house / Наружняя отделка дома  ?',
    ('VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
       'Stone', 'ImStucc', 'CBlock'))
    res_df['Exterior1st'] = Exterior1st
    
    ExterCond = st.selectbox(
    'Present condition of the material on the exterior / Текущее состояние снаружи ?',
    ('TA', 'Gd', 'Fa', 'Po', 'Ex'))
    res_df['ExterCond'] = ExterCond
    
    Foundation = st.selectbox(
    'Type of foundation / Тип фундамента ?',
    ('PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'))
    res_df['Foundation'] = Foundation
    
    BsmtQual = st.selectbox(
    'Height of the basement / Высота подвала  ?',
    ('Gd', 'TA', 'Ex','Fa'))
    res_df['BsmtQual'] = BsmtQual
    
    
    TotalBsmtSF = st.number_input('Total square feet of basement area / Площадь подвала ?')
    res_df['TotalBsmtSF'] = TotalBsmtSF
    
    Heating = st.selectbox(
    'Type of heating / Тип отопления ?',
    ('GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'))
    res_df['Heating'] = Heating
    
    CentralAir = st.selectbox(
    'Central air conditioning / Центральный кондиционер ?',
    ('Yes', 'No'))
    CentralAir = 'Y' if CentralAir == 'Yes' else 'N'
    res_df['CentralAir'] = CentralAir
    
    Electrical = st.selectbox(
    'Electrical system / электрическая система ?',
    ('SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'))
    res_df['Electrical'] = Electrical
    
    stFlrSF = st.number_input('First Floor square feet / Площадь 1-го этажа ?')
    res_df['1stFlrSF'] = stFlrSF
    
    ndFlrSF = st.number_input('Second floor square feet / Площадь 2-го этажа ?')
    res_df['2ndFlrSF'] = ndFlrSF
    
    GarageFinish = st.selectbox(
    'Interior finish of the garage / Внутрення отделка гаража ?',
    ('RFn', 'Unf', 'Fin'))
    res_df['GarageFinish'] = GarageFinish
    
    GarageCars = st.slider('Size of garage in car capacity / Вместительность гаража',1,10)
    res_df['GarageCars'] = GarageCars
    
    PavedDrive = st.selectbox(
    'Paved driveway / Асфальтированная подъездная дорога  ?',
    ('Yes', 'No'))
    PavedDrive = 'Y' if PavedDrive == 'Yes' else 'N'
    res_df['PavedDrive'] = PavedDrive
    
    ScreenPorch = st.number_input('Screen porch area in square feet / Площадь веранды  ?')
    res_df['ScreenPorch'] = ScreenPorch
    
    PoolArea = st.number_input('Pool area in square feet / Площадь бассейна  ?')
    res_df['PoolArea'] = PoolArea
    
res = ml_pipeline.predict(res_df)


if st.button('Расчитать цену'):
    st.title('Предсказанная цена')
    st.header("Результат: " + str(round(*np.exp(res), 2)) + " $" if GrLivArea != 0 else 'Данные не заполнены не полностью')
    
    
    
    


    








