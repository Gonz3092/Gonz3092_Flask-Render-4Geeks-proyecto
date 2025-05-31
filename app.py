import os
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Define la ruta completa al modelo usando os.path.join
# Esto hace que la ruta sea compatible tanto en desarrollo local como en Render
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav')

# Cargar el modelo de Decision Tree
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print("Modelo cargado exitosamente!")
except FileNotFoundError:
    print(f"Error: El archivo del modelo no se encontró en {MODEL_PATH}")
    model = None # Asegúrate de manejar este caso si el modelo no carga
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

@app.route('/')
def home():
    # Puedes crear una página de inicio simple o una documentación de la API
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'El modelo no está disponible. Contacte al administrador.'}), 500

    try:
        # Obtener los datos del paciente desde la solicitud JSON
        data = request.get_json(force=True)

        # Extraer las variables en el orden correcto que espera tu modelo
        # Asegúrate de que las claves en el JSON coincidan exactamente
        pregnancies = data['Pregnancies']
        glucose = data['Glucose']
        blood_pressure = data['BloodPressure']
        skin_thickness = data['SkinThickness']
        insulin = data['Insulin']
        bmi = data['BMI']
        diabetes_pedigree_function = data['DiabetesPedigreeFunction']
        age = data['Age']

        # Crear un array numpy para la predicción
        # Es crucial que el orden de las características sea el mismo que se usó para entrenar el modelo
        features = [[pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree_function, age]]

        # Realizar la predicción
        prediction = model.predict(features)
        
        # Obtener la probabilidad de cada clase (opcional, pero útil para modelos de clasificación)
        prediction_proba = model.predict_proba(features)

        # El resultado de la predicción es un array, toma el primer elemento
        outcome = int(prediction[0])
        
        # Formatear la respuesta
        result = {
            'prediction': outcome,
            'prediction_label': 'Positivo en Diabetes' if outcome == 1 else 'Negativo en Diabetes',
            'probability_negative': prediction_proba[0][0], # Probabilidad de 0 (Negativo)
            'probability_positive': prediction_proba[0][1]  # Probabilidad de 1 (Positivo)
        }

        return jsonify(result)

    except KeyError as e:
        return jsonify({'error': f'Falta el parámetro: {e}. Asegúrate de enviar todas las variables.'}), 400
    except Exception as e:
        return jsonify({'error': f'Ocurrió un error inesperado: {e}'}), 500

if __name__ == '__main__':
    # Para desarrollo local, usa debug=True
    # En producción (Render), Gunicorn o un servidor WSGI se encargará de esto
    app.run(debug=True)