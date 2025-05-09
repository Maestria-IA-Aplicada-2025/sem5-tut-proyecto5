import pandas as pd

# Cargar el archivo CSV desde el archivo descargado localmente
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)  # Asegúrate de poner la ruta correcta

# Guardar el DataFrame en un archivo Excel
df.to_excel('titanic_data.xlsx', index=False)
