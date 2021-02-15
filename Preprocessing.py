# Base imports
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset
dataset = pd.read_csv(r"04_dados_exercicio.csv")

# Variáveis independentes (features) e dependentes (classes)
features = dataset.iloc[:, 1:-1].values
classe = dataset.iloc[:, -1].values

# Substituição de dados faltantes
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
features[:, 2:4] = imputer.fit_transform(features[:, 2:4])

# Codificação de variáveis categóricas por OneHotEncoder
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
features = columnTransformer.fit_transform(features)

# Codificação da variável dependente por rótulo
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)

# Separação do conjunto em Teste e Treinamento (por um fator de 85%)
features_treinamento, features_teste = train_test_split(features, test_size=0.15, random_state=1)
classe_treinamento, classe_teste = train_test_split(classe, test_size=0.15, random_state=1)

# Normalização de Temperatura e Umidade
scaler = StandardScaler()
features_treinamento[:, 3:5] = scaler.fit_transform(features_treinamento[:, 3:5])
features_teste[:, 3:5] = scaler.fit_transform(features_teste[:, 3:5])

# Printer -----
print('\n=-=-=-=[ Features ]=-=-=-=')
print(' -> Treinamento:')
print(features_treinamento)
print(' -> Teste:')
print(features_teste)
print('\n=-=-=-=[  Classe  ]=-=-=-=')
print(' -> Treinamento:')
print(classe_treinamento)
print(' -> Teste:')
print(classe_teste)