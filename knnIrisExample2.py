import Orange
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1️⃣ Cargar datos y entrenar
# =========================
data = Orange.data.Table("iris")

knn = Orange.classification.KNNLearner(n_neighbors=3)
model = knn(data)

# =========================
# 2️⃣ Crear nueva flor y clasificar
# =========================
nueva_data = Orange.data.Table.from_list(
    data.domain,
    [[5.1, 3.5, 1.4, 0.2, None]]
)

pred = model(nueva_data)
indice = int(pred[0])
clase_predicha = data.domain.class_var.values[indice]

print("Índice:", indice)
print("Clase predicha:", clase_predicha)

# =========================
# 3️⃣ GRÁFICA 1
# Scatter real por clase
# =========================
X = data.X
y = data.Y

plt.figure()
for clase in range(len(data.domain.class_var.values)):
    plt.scatter(
        X[y == clase, 2],   # Petal length
        X[y == clase, 3],   # Petal width
        label=data.domain.class_var.values[clase]
    )

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Dataset - Clases reales")
plt.legend()
plt.show()


# =========================
# 4️⃣ GRÁFICA 2
# Predicciones del modelo
# =========================
predicciones = model(data)

plt.figure()
for clase in range(len(data.domain.class_var.values)):
    plt.scatter(
        X[np.array(predicciones) == clase, 2],
        X[np.array(predicciones) == clase, 3],
        label=f"Pred: {data.domain.class_var.values[clase]}"
    )

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Clasificación usando KNN")
plt.legend()
plt.show()