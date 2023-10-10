# Importando as bibliotecas necessárias
import numpy as np  # Biblioteca para operações matemáticas
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector  # Biblioteca para manipulação de dados
import tensorflow as tf 

# Carregando o conjunto de dados
dataset = pd.read_csv("Churn_Modelling.csv")

# Separando as características/features (X) e a variável dependente (y)
x = dataset.iloc[:, 3:-1].values  # Pegando todas as colunas, exceto a última
x = np.array(x)  # Convertendo o resultado em uma matriz numpy
y = dataset.iloc[:, -1].values  # Pegando apenas a última coluna
y = np.array(y)  # Convertendo o resultado em uma matriz numpy

# Iniciando a ANN
ann = tf.keras.models.Sequential()

# Adicionando a layer de entrada e a primeira hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adicionando a segunda hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adicionando a layer de saída
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))