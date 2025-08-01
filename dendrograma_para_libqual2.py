# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 04:46:20 2025

@author: patom
"""
# PARA HACER UN DENDROGRAMA EN LA BASE DE DATOS SÓLO DEBEN ESTASR LAS VARIABLES 
# MÉTRICAS QUE SE UTILIZARÁN EN EL ANÁLSIS, SE DEBE BORRAR TODAS LAS VARIABLES 
# QUE NO SE OCUPEN

# ESTE DENDROGRAMA LO HICE CON STREAMLIT

# !pip install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Creación de un DataFrame de ejemplo ---
# REEMPLAZA ESTA SECCIÓN CON TU PROPIO DATAFRAME 'df'
# Asegúrate de que tu DataFrame solo contenga las columnas numéricas
# que quieres usar para el clustering.
# np.random.seed(42) # Para reproducibilidad

df = pd.read_excel('BD LIBQUAL DENDOG.xlsx')

st.header("Base de datos")
df


# 1. Estandarizar los datos (Paso recomendado)
# Es importante escalar los datos para que las variables con rangos más grandes
# no dominen el análisis de distancia.
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convertir de nuevo a DataFrame para mantener las etiquetas
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)


# 2. Calcular el 'linkage' (enlace jerárquico)
# El método 'ward' es una opción popular que minimiza la varianza
# de los clústeres que se fusionan.
# Otros métodos comunes son 'complete', 'average', 'single'.
linked = linkage(df_scaled, method='ward')


# 3. Graficar el dendrograma

# Crea una figura y un objeto de ejes de forma explícita.
# Esto es la clave para que funcione bien con Streamlit.
fig, ax = plt.subplots(figsize=(14, 8))

# Pasa el objeto 'ax' para que el dendrograma se dibuje en él.
dendrogram(
    linked,
    orientation='top',
    labels=df_scaled.index,  # Usa los índices del DataFrame como etiquetas
    distance_sort='descending',
    leaf_rotation=90,  # Rota las etiquetas para que sean legibles
    leaf_font_size=10,  # Tamaño de la fuente para las etiquetas
    ax=ax # ESPECIFICA LOS EJES DONDE DIBUJAR EL DENDROGRAMA
)

# Añadir títulos y etiquetas usando los métodos del objeto 'ax'
ax.set_title('Dendrograma de Clustering Jerárquico', fontsize=16)
ax.set_xlabel('Muestras', fontsize=12)
ax.set_ylabel('Distancia (Ward)', fontsize=12)
ax.grid(axis='y') # Mantiene la cuadrícula solo en el eje y

# Ajusta el diseño de la figura
fig.tight_layout()

# Guarda el gráfico en .png
fig.savefig('dendrograma_bibqual.png', dpi=300, bbox_inches='tight')

# Muestra el gráfico en Streamlit, pasando el objeto 'fig' explícitamente
st.pyplot(fig)