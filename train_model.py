
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Cargar el dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Preprocesamiento
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]  # Usamos algunas características
data['Age'].fillna(data['Age'].mean(), inplace=True)  # Rellenamos valores nulos en 'Age'

# Convertimos las variables categóricas en números
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Dividir los datos en características y etiquetas
X = data.drop('Survived', axis=1)
y = data['Survived']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar el modelo y el vectorizador
with open('model_titanic.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo entrenado y guardado correctamente.")
