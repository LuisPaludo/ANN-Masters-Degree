import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import joblib

# Meu modelo
versao = '4'
# Carregar o arquivo CSV
dados = pd.read_csv(f'data/dados_treinamento.csv')

# Selecionar features e labels
features = dados[['e_iq', 'e_id', 'iqs', 'ids']]
labels = dados[['Vq', 'Vd']]

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

mat_dict = {
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max
}

scipy.io.savemat(f'saved_model/data/my_model_{versao}/scaler_values.mat', mat_dict)
joblib.dump(scaler_features,f'saved_model/data/my_model_{versao}/scaler_features.save')
joblib.dump(scaler_labels,f'saved_model/data/my_model_{versao}/scaler_labels.save')

# Dividir os dados em treinamento e teste
features_train, features_test, labels_train, labels_test = train_test_split(
    features_scaled, labels_scaled, test_size=0.2, random_state=42
)

# Definir a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
    #  tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
     tf.keras.layers.Dense(2)  # Nenhuma função de ativação para a camada de saída em um problema de regressão
])

# Compilar o modelo
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# Resumo da arquitetura
model.summary()

# Treinar o modelo
history = model.fit(
    features_train, labels_train,
    epochs=100,  # número de vezes que o modelo verá todo o conjunto de treinamento
    batch_size=256,  # número de amostras processadas antes de atualizar os pesos do modelo
    validation_data=(features_test, labels_test),
    verbose=1  # type: ignore # mostra o progresso do treinamento
)

# Plotar histórico de treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend()
plt.title('MAE Evolution')

plt.show()

# Avaliar o modelo no conjunto de teste
loss, mae = model.evaluate(features_test, labels_test, verbose=0) # type: ignore

print(f"Erro Quadrado Médio (MSE) no conjunto de teste: {loss:.5f}")
print(f"Erro Absoluto Médio (MAE) no conjunto de teste: {mae:.5f}")

# Fazer previsões com o conjunto de teste
predictions = model.predict(features_test)

# Desnormalizar as previsões e os labels reais
predictions_real = scaler_labels.inverse_transform(predictions)
labels_test_real = scaler_labels.inverse_transform(labels_test)

# Plotar algumas previsões vs. valores reais
plt.figure(figsize=(12, 5))
plt.plot(labels_test_real[:100, 0], label="Real Vq", color='blue')
plt.plot(predictions_real[:100, 0], label="Predicted Vq", color='red', linestyle='dashed')
plt.legend()
plt.title("Real vs. Predicted Vq for the first 100 test samples")
plt.show()

# Salvar o modelo
model.save_weights(f'saved_model/data/my_model_{versao}/model_weights.h5')
model.save(f'saved_model/my_model_{versao}')








