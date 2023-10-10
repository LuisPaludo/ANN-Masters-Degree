# Importando as bibliotecas necessárias
import numpy as np  # Biblioteca para operações matemáticas
import pandas as pd  # Biblioteca para manipulação de dados
import matplotlib.pyplot as plt  # Biblioteca para visualização de dados
from sklearn.impute import SimpleImputer  # Ferramenta para tratar valores faltantes
from sklearn.compose import (
    ColumnTransformer,
)  # Ferramenta para aplicar transformações em colunas específicas
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
)  # Ferramentas para codificação e padronização
from sklearn.model_selection import (
    train_test_split,
)  # Ferramenta para dividir conjuntos de dados

# Carregando o conjunto de dados
dataset = pd.read_csv("data_pre_processing/Data.csv")

# Separando as características/features (X) e a variável dependente (y)
x = dataset.iloc[:, :-1].values  # Pegando todas as colunas, exceto a última
x = np.array(x)  # Convertendo o resultado em uma matriz numpy
y = dataset.iloc[:, -1].values  # Pegando apenas a última coluna
y = np.array(y)  # Convertendo o resultado em uma matriz numpy

# Inicializando o SimpleImputer para tratar valores faltantes
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Aplicando o imputer nas colunas com índices 1 e 2 para substituir valores faltantes pela média da coluna
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])

# Inicializando o ColumnTransformer para codificar a primeira coluna usando OneHotEncoder
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)

# Aplicando a transformação e atualizando x
x = ct.fit_transform(x)

# Inicializando o LabelEncoder para codificar a variável dependente y
le = LabelEncoder()
y = le.fit_transform(y)

# Dividindo o conjunto de dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Inicializando o StandardScaler para padronização
sc = StandardScaler()

# Aplicando o StandardScaler para padronizar as características a partir da quarta coluna (índice 3) em diante
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
