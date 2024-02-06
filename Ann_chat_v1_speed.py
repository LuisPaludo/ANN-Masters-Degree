# Primeiro Modelo pera estimação de velocidade
# Baseado no Ann_chat_V7

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import levenberg_marquardt as lm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import joblib
import os

# Meu modelo
versao = "2"

# Verifique se o diretório existe. Se não, crie-o.
dir_path = f"saved_model/data/speed_estimation/my_model_{versao}/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Verifique se o diretório existe. Se não, crie-o.
dir_path = f"saved_model/speed_estimation/my_model_{versao}/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Carregar e randomizar o arquivo de treinamento
dados = pd.read_csv(f"data/dados_treinamento_4.csv")

# Para o conjunto de treinamento
features = dados[["iqs", "ids", "Vq", "Vd"]]
labels = dados[["wr"]]

# Criar um scaler para as features e outro para os labels
scaler_features = MinMaxScaler()
scaler_labels = MinMaxScaler()

# Fit e transform os dados com os scalers
features_scaled = scaler_features.fit_transform(features)
labels_scaled = scaler_labels.fit_transform(labels)

x_min = scaler_features.data_min_
x_max = scaler_features.data_max_
y_min = scaler_labels.data_min_
y_max = scaler_labels.data_max_

mat_dict = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

scipy.io.savemat(
    f"saved_model/data/speed_estimation/my_model_{versao}/scaler_values.mat", mat_dict
)
joblib.dump(
    scaler_features,
    f"saved_model/data/speed_estimation/my_model_{versao}/scaler_features.save",
)
joblib.dump(
    scaler_labels,
    f"saved_model/data/speed_estimation/my_model_{versao}/scaler_labels.save",
)

# Dividir os dados em treinamento e teste
features_train, features_test, labels_train, labels_test = train_test_split(
    features_scaled, labels_scaled, test_size=0.2, random_state=42
)

# Definir a arquitetura da rede neural
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# Resumo da arquitetura
model.summary()

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="loss")  # type: ignore
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.1, patience=5, monitor="loss", cooldown=5, min_delta=1e-5
)

# Compilar o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae"],
)

model_wrapper = lm.ModelWrapper(tf.keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss=lm.MeanSquaredError(), metrics=["mae"],
)

history = model_wrapper.fit(
    x=features_train,
    y=labels_train,
    epochs=2,
    batch_size=256,
    validation_data=(features_test, labels_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
)

# Salvar o modelo
model.save_weights(
    f"saved_model/data/speed_estimation/my_model_{versao}/model_weights.h5"
)
model.save(f"saved_model/speed_estimation/my_model_{versao}")


# history = model.fit(
#     features_train, labels_train,
#     epochs=150,
#     batch_size=256*2,
#     validation_data=(features_test, labels_test),
#     callbacks=[early_stopping, reduce_lr],
#     verbose=1 #type: ignore
# )

# Plotar histórico de treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Evolution")

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.legend()
plt.title("MAE Evolution")

plt.show()

# Avaliar o modelo no conjunto de teste
loss, mae = model.evaluate(features_test, labels_test, verbose=0)  # type: ignore

print(f"Erro Quadrado Médio (MSE) no conjunto de teste: {loss:.5f}")
print(f"Erro Absoluto Médio (MAE) no conjunto de teste: {mae:.5f}")

# Fazer previsões com o conjunto de teste
predictions = model.predict(features_test)

# Desnormalizar as previsões e os labels reais
predictions_real = scaler_labels.inverse_transform(predictions)
labels_test_real = scaler_labels.inverse_transform(labels_test)

# Plotar algumas previsões vs. valores reais
plt.figure(figsize=(12, 5))
plt.plot(labels_test_real[:100, 0], label="Real wr", color="blue")
plt.plot(
    predictions_real[:100, 0], label="Predicted wr", color="red", linestyle="dashed"
)
plt.legend()
plt.title("Real vs. Predicted Vq for the first 100 test samples")
plt.show()

