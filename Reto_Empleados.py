# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:06:26 2025

@author: jrvarela
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Lee el archivo CSV llamado empleadosRETO.csv y coloca los datos en un frame de Pandas llamado EmpleadosAttrition.
Empleados_Attrition = pd.read_csv(r'C:\Users\jrvarela.CNE\OneDrive - Comisión Nacional de Energía\JOREVA - GLP\TEC\empleadosReto.csv')
Empleados_Attrition.head()

Empleados_Attrition.dtypes


# Elimina las columnas que, con alta probabilidad (estimada por ti), no tienen relación alguna con la salida. Hay algunas columnas que contienen información que no ayuda a definir el desgaste de un empleado
Empleados_Attrition = Empleados_Attrition.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis = 1)

# Analiza la información proporcionada, si detectaste que no se cuenta con los años que el empelado lleva en la compañía 
# y parece ser un buen dato. Dicha cantidad se puede calcular con la fecha de contratación ‘HiringDate’.
#Crea una columna llamada Year y obtén el año de contratación del empleado a partir de su fecha ‘HiringDate’. No se te olvide que debe ser un entero.
#Crea una columna llamada YearsAtCompany que contenga los años que el empleado lleva en la compañía hasta el año 2018. Para su cálculo, usa la variable Year que acabas de crear.

Empleados_Attrition['HiringDate'] = pd.to_datetime(Empleados_Attrition['HiringDate'], errors='coerce')
Empleados_Attrition = Empleados_Attrition.dropna(subset=['HiringDate'])  

Empleados_Attrition['Year'] = pd.DatetimeIndex(Empleados_Attrition.HiringDate).year
Empleados_Attrition['YearsAtCompany'] = 2018 - Empleados_Attrition['Year']


#La DistanceFromHome está dada en kilómetros, pero tiene las letras “km” al final y así no puede ser entera.
#Renombra la variable DistanceFromHome a DistanceFromHome_km.
#Crea una nueva variable DistanceFromHome que sea entera, es decir, solo con números.

Empleados_Attrition.rename(columns={'DistanceFromHome': 'DistanceFromHome_km'}, inplace=True)
Empleados_Attrition['DistanceFromHome'] = Empleados_Attrition['DistanceFromHome_km'].str.replace(' km', '', regex=False).astype(int)


# Borra las columnas Year, HiringDate y DistanceFromHome_km debido a que ya no son útiles.
Empleados_Attrition = Empleados_Attrition.drop(columns=['Year', 'HiringDate', 'DistanceFromHome_km'])

# Aprovechando los ajustes que se están haciendo, la empresa desea saber si todos los departamentos tienen un ingreso promedio similar.
# Genera una nuevo frame llamado SueldoPromedioDepto que contenga el MonthlyIncome promedio por departamento de los empleados
# y colócalo en una variable llamada SueldoPromedio. Esta tabla solo es informativa, no la vas a utilizar en el set de datos que estás construyendo.

SueldoPromedioDepto = Empleados_Attrition.groupby('Department')[['MonthlyIncome']].mean().rename(columns= {'MonthlyIncome': 'SueldoPromedio'})

# La variable MonthlyIncome tiene un valor numérico muy grande comparada con las otras variables. Escala dicha variable para que tenga un valor entre 0 y 1. 

Empleados_Attrition['MonthlyIncome'] = (Empleados_Attrition['MonthlyIncome'] - Empleados_Attrition['MonthlyIncome'].min()) / (Empleados_Attrition['MonthlyIncome'].max() - Empleados_Attrition['MonthlyIncome'].min())

# odo parece indicar que las variables categóricas que quedan sí son importantes para obtener la variable de salida. Convierte todas las variables categóricas que quedan a numéricas

categorical_vars = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Attrition']
EA_encoded = pd.get_dummies(Empleados_Attrition, columns=categorical_vars, drop_first=True)


# Ahora debes hacer la evaluación de las variables para quedarte con las mejores. Calcula la correlación lineal de cada una de las variables
# Ahora debes hacer la evaluación de las variables para quedarte con las mejores. Calcula la correlación lineal de cada una de las variables con respecto al Attrition. con respecto al Attrition.

correlaciones = EA_encoded.corr()['Attrition_Yes'].abs()
relevant_vars = correlaciones[correlaciones >= 0.1].index.tolist()
EmpleadosAttritionFinal = EA_encoded[relevant_vars]

# Crea una nueva variable llamada EmpleadosAttritionPCA formada por los componentes principales del frame EmpleadosAttritionFinal. Recuerda que el resultado del proceso PCA es un numpy array, por lo que, para hacer referencia a una columna, por ejemplo, la 0, puedes usar la instrucción EmpleadosAttritionPCA[:,0]).
# Agrega el mínimo número de Componentes Principales en columnas del frame EmpleadosAttritionPCA que logren explicar el 80% de la varianza, al frame EmpleadosAttritionFinal. Puedes usar la instrucción assign, columna por columna, llamando a cada unaC0, C1, etc., hasta las que vayas a agregar.

pca = PCA()
EmpleadosAttritionPCA = pca.fit_transform(EmpleadosAttritionFinal)

# Obtener número mínimo de componentes necesarios para explicar 80%
explained_var = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(explained_var >= 0.8) + 1

# Agregar columnas C0, C1,... al dataframe
for i in range(num_components):
    EmpleadosAttritionFinal[f'C{i}'] = EmpleadosAttritionPCA[:, i]

# Guardar CSV

EmpleadosAttritionFinal.to_csv(r'C:\Users\jrvarela.CNE\OneDrive - Comisión Nacional de Energía\JOREVA - GLP\TEC\EmpleadosAttritionFinal.csv', index=False)