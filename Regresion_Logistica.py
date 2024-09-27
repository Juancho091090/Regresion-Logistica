import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cargar datos
data = pd.read_csv('student_grades.csv')

# Visualización y revisión de los datos
print(data.head())
print(data.info())
print(data.describe())

# Crear la columna 'Passed' con 1 si el estudiante faltó menos de 5 clases, 0 en caso contrario
data['Passed'] = np.where(data['Absences'] < 5, 1, 0)

# Seleccionar características (MathScore, ReadingScore, WritingScore) y la variable dependiente (Passed)
X = data[['MathScore', 'ReadingScore', 'WritingScore']]
y = data['Passed']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nExactitud del modelo:")
print(accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Visualización opcional de correlaciones
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
