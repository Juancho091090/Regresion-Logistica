import os

# Establecer el número máximo de núcleos para evitar problemas con joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
    
# Importar bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generar datos de ejemplo
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Configuración e implementación de K-means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, algorithm='elkan')
kmeans.fit(X)
labels = kmeans.predict(X)

# Visualización de resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroides')
plt.title("Clusters encontrados por K-means")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()
