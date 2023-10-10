import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tf2onnx
import numpy as np
import scipy.io

# Carregar os dados
dados = pd.read_csv("dados_treinamento.csv")

# Separar os dados em entradas (X) e saídas (y)
X = dados.drop(["U_iq", "U_id"], axis=1)
y = dados[["U_iq", "U_id"]]

# Dividir os dados em conjuntos de treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train_normalized = scaler_X.transform(X_train)
X_val_normalized = scaler_X.transform(X_val)
y_train_normalized = scaler_y.transform(y_train)
y_val_normalized = scaler_y.transform(y_val)

# Definir a arquitetura da rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(13,)),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# Sumário do modelo
model.summary()

# Treinar o modelo
history = model.fit(X_train_normalized, y_train_normalized, 
                    epochs=1, 
                    batch_size=32, 
                    validation_data=(X_val_normalized, y_val_normalized))

val_loss, val_mae = model.evaluate(X_val_normalized, y_val_normalized)
print("Validation MAE:", val_mae)

predictions = model.predict(X_val_normalized)

model.save_weights('model_weights.h5')

model.save('saved_model/my_model')

# Converta o modelo TensorFlow para ONNX
model = tf.keras.models.load_model('saved_model/my_model')
# Salvar o modelo ONNX
tf2onnx.convert.from_keras(model, output_path='model.onnx')

# 1. Extraia os parâmetros de média e desvio padrão dos scalers e garanta que eles são ndarrays
mean_X = np.array(scaler_X.mean_)
std_X = np.array(scaler_X.scale_)
mean_y = np.array(scaler_y.mean_)
std_y = np.array(scaler_y.scale_)

# Salvar em formato .csv
np.savetxt("mean_X.csv", mean_X, delimiter=",")
np.savetxt("std_X.csv", std_X, delimiter=",")
np.savetxt("mean_y.csv", mean_y, delimiter=",")
np.savetxt("std_y.csv", std_y, delimiter=",")

# Alternativamente, salvar em formato .mat
scipy.io.savemat("scaler_params.mat", {
    'mean_X': mean_X,
    'std_X': std_X,
    'mean_y': mean_y,
    'std_y': std_y
})
