# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 07:48:57 2025

@author: patom
"""

# ANALISIS DE REGRESIÓN Y CORRELACIÓN MULTIPLE

# !pip install seaborn
import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

# cargaremos datos de la BD

df = pd.read_excel('BD ENCUESTA SATISF LAB.xlsx')


print("DataFrame cargado:")
print(df.head(3))
print("\n")


print('===============================')

print('Filas, Columnas')
print(df.shape)

print('===============================')

# --- 2. Análisis de Correlación Múltiple ---
# La correlación mide la fuerza y dirección de una relación lineal entre dos variables.
# Una matriz de correlación nos mostrará la correlación entre todas las parejas de variables.

st.header("Matriz de Correlación:")
correlation_matrix = df.corr()
st.write(correlation_matrix)
st.write("\n")

# Visualización de la matriz de correlación con un Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.header('Matriz de Correlación')

# plt.show()
st.pyplot()

# --- 3. Análisis de Regresión Múltiple ---

# La regresión múltiple modela la relación entre una variable dependiente (Y)
# y dos o más variables independientes (X).

# Define la variable dependiente (Y) y las variables independientes (X)
# Asegúrate de que los nombres de las columnas coincidan con tu DataFrame.
# En este ejemplo, 'Y' es la dependiente, 'X1', 'X2', 'X3' son las independientes.

# Método 1: Usando statsmodels con R-style formulas (recomendado para simplicidad)


# La fórmula es 'variable_dependiente ~ variable_independiente_1 + variable_independiente_2 + ...'

# ________________________________________________________________

# INGRESAR AQUÍ EL NOMBRE DE LAS VARIABLES QUE QUEREMOS ANALIZAR

# formula = 'Y ~ X1 + X2 + X3'

formula = 'ausent ~ p7relj + p1remun + p4ho + edad + p3relc'


model = ols(formula, data=df).fit()

st.write("Resultados de la Regresión Múltiple (Método R-style formula):")
st.write(model.summary())
st.write("\n")

# Interpretación de los resultados clave del modelo:
# - R-squared (R cuadrado): Proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes.
# - Adj. R-squared (R cuadrado ajustado): Versión ajustada del R cuadrado, útil para comparar modelos con diferente número de predictores.
# - F-statistic (Estadístico F) y Prob (F-statistic): Evalúa la significancia general del modelo de regresión. Un p-valor bajo (< 0.05) sugiere que el modelo es estadísticamente significativo.
# - Coef (Coeficientes): El cambio promedio en la variable dependiente por cada unidad de cambio en la variable independiente correspondiente, manteniendo las demás variables constantes.
# - std err (Error estándar): Medida de la precisión de los coeficientes.
# - t (Valor t) y P>|t| (p-valor): Evalúa la significancia individual de cada variable independiente. Un p-valor bajo (< 0.05) sugiere que la variable es un predictor significativo.
# - [0.025, 0.975] (Intervalo de Confianza): El rango dentro del cual se espera que caiga el verdadero valor del coeficiente.


# no ------------------------------------------------------------------------------------------------
# Método 2: Usando statsmodels con matrices X e Y explícitas
# Este método es útil si prefieres construir tus matrices de características manualmente.
# X = df[['X1', 'X2', 'X3']] # Variables independientes
# Y = df['Y'] # Variable dependiente

# # Añadir una constante al modelo para el intercepto (término independiente)
# X = sm.add_constant(X)

# model_explicit = sm.OLS(Y, X).fit()

# print("Resultados de la Regresión Múltiple (Método X e Y explícitas):")
# print(model_explicit.summary())
# print("\n")




# --- 4. Predicciones (Opcional) ---


# Una vez que tienes un modelo, puedes usarlo para hacer predicciones.

# Crear nuevos datos para la predicción (asegúrate de que tengan las mismas columnas X)
#new_data = pd.DataFrame({
#    'X1': [12, 13],
#    'X2': [11, 12],
#    'X3': [8, 9]
# })

# Para el método de fórmula, las predicciones son directas
# predictions = model.predict(new_data)
# print("Predicciones para nuevos datos:")
# print(predictions)
# print("\n")

# Para el método explícito, también necesitarías añadir la constante si la añadiste en el entrenamiento
# new_data_explicit = sm.add_constant(new_data)
# predictions_explicit = model_explicit.predict(new_data_explicit)
# print("Predicciones para nuevos datos (explícito):")
# print(predictions_explicit)
# print("\n")

# --- 5. Residuos (Opcional) ---
# Los residuos son la diferencia entre los valores observados y los valores predichos.
# Analizar los residuos es importante para verificar los supuestos de la regresión.

residuals = model.resid
st.write("Primeros 5 residuos:")
st.write(residuals.head())

# Visualización de los residuos para verificar la normalidad y la homocedasticidad
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True)
plt.title('Distribución de los Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
plt.scatter(model.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Residuos vs. Valores Predichos')
# plt.show()

st.pyplot()