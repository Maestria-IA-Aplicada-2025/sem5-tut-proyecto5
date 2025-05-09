
from flask import Flask, render_template, request
import pickle
import numpy as np

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado
with open('model_titanic.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar el vectorizador para transformar los datos de texto
#with open('vectorizer.pkl', 'rb') as file:
   # vectorizer = pickle.load(file)

# Ruta principal para mostrar el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para manejar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtenemos el mensaje del formulario
    pclass = int(request.form['Pclass'])
    sex = 1 if request.form['Sex'] == 'male' else 0  # Convertir 'male' a 1 y 'female' a 0
    age = float(request.form['Age'])
    sibsp = int(request.form['SibSp'])
    parch = int(request.form['Parch'])
    fare = float(request.form['Fare'])
    
    # Predecir la supervivencia usando el modelo
    prediction = model.predict([[pclass, sex, age, sibsp, parch, fare]])
    
    # Devolver el resultado
    result = "Survived" if prediction[0] == 1 else "Not Survived"
    return render_template('result.html', prediction=result)

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
