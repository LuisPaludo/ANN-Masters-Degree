import tensorflow as tf
import joblib

model = tf.keras.models.load_model('saved_model/my_model_2')
scaler_features = joblib.load('saved_model/data/my_model_2/scaler_features.save')
scaler_labels = joblib.load('saved_model/data/my_model_2/scaler_labels.save')

# teste
test_input_data_normalized = [[0.606091473302615, 0.550267092675244, 0.653029482734828, 0.445701097934596]]
test_input_data = [[0, 0.00718038556095148, 0, 0.103371297347282]]
real_test_input_data_normalized = scaler_features.transform(test_input_data)

real_test_predictions_normalized = model.predict(real_test_input_data_normalized)
real_test_predictions = scaler_labels.inverse_transform(real_test_predictions_normalized)

test_predictions_my_normalized = model.predict(test_input_data_normalized)
test_predictions = scaler_labels.inverse_transform(test_predictions_my_normalized)

test_output_data_normalized = [[0.4969884, 0.5555388]]
real_test_output_data = scaler_labels.inverse_transform(test_output_data_normalized)
test_output_data = [-1.081792, -8.412831]

print("Entrada Normalizada")
print(real_test_input_data_normalized)
print("Minha Normalização")
print(test_input_data_normalized)

print("Predição")
print(real_test_predictions)
print("Minha Predição")
print(test_output_data)
print("Predição com meus dados normalizados")
print(test_predictions)

print('Meu Resultado Normalizado')
print(real_test_output_data)
print('Meu Resultado')
print(test_output_data)
