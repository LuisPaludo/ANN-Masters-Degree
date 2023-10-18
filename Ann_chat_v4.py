# Terceiro Modelo com bons resultados
# Diminuição da complexidade da rede neural
# Saindo de uma rede Deep para uma rede Shallow
# Número de hidden layers reduzidos: 4 -> 2
# Randomização dos dados de treinamento e validação

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import joblib
import os

# Meu modelo
versao = '7'

# Verifique se o diretório existe. Se não, crie-o.
dir_path = f'saved_model/data/my_model_{versao}/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Verifique se o diretório existe. Se não, crie-o.
dir_path = f'saved_model/my_model_{versao}/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Carregar e randomizar o arquivo de treinamento
dados = pd.read_csv(f'data/dados_treinamento_2.csv')
dados = dados.sample(frac=1).reset_index(drop=True)

# Carregar e randomizar o arquivo de validação
dados_validacao = pd.read_csv('data/dados_validacao_1.csv')
dados_validacao = dados_validacao.sample(frac=1).reset_index(drop=True)

# Para o conjunto de treinamento
features_train = dados[['e_iq', 'e_id', 'iqs', 'ids']]
labels_train = dados[['Vq', 'Vd']]

# Para o conjunto de validação
features_validacao = dados_validacao[['e_iq', 'e_id', 'iqs', 'ids']]
labels_validacao = dados_validacao[['Vq', 'Vd']]

# Criar um scaler para as features e outro para os labels
scaler_features = MinMaxScaler()
scaler_labels = MinMaxScaler()

# Ajustando os scalers apenas com os dados de treinamento
scaler_features.fit(features_train)
scaler_labels.fit(labels_train)

# Transformando os conjuntos de treinamento e validação
features_train_scaled = scaler_features.transform(features_train)
labels_train_scaled = scaler_labels.transform(labels_train)

features_validacao_scaled = scaler_features.transform(features_validacao)
labels_validacao_scaled = scaler_labels.transform(labels_validacao)

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

# Definir a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  
])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# Resumo da arquitetura
model.summary()

history = model.fit(
    features_train_scaled, labels_train_scaled,
    epochs=150,
    batch_size=256,
    validation_data=(features_validacao_scaled, labels_validacao_scaled),
    callbacks=[early_stopping, reduce_lr],
    verbose=1 #type: ignore
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
loss, mae = model.evaluate(features_validacao_scaled, labels_validacao_scaled, verbose=0) # type: ignore

print(f"Erro Quadrado Médio (MSE) no conjunto de teste: {loss:.5f}")
print(f"Erro Absoluto Médio (MAE) no conjunto de teste: {mae:.5f}")

# Fazer previsões com o conjunto de teste
predictions = model.predict(features_validacao_scaled)

# Desnormalizar as previsões e os labels reais
predictions_real = scaler_labels.inverse_transform(predictions)
labels_test_real = scaler_labels.inverse_transform(labels_validacao_scaled)

# Plotar algumas previsões vs. valores reais
plt.figure(figsize=(12, 5))
plt.plot(labels_test_real[:500, 0], label="Real Vq", color='blue')
plt.plot(predictions_real[:500, 0], label="Predicted Vq", color='red', linestyle='dashed')
plt.legend()
plt.title("Real vs. Predicted Vq for the first 100 test samples")
plt.show()

# Salvar o modelo
model.save_weights(f'saved_model/data/my_model_{versao}/model_weights.h5')
model.save(f'saved_model/my_model_{versao}')








