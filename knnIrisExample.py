import Orange

# Cargar el conjunto de datos Iris
data = Orange.data.Table("iris")

# Crear el modelo KNN con k=3
knn = Orange.classification.KNNLearner(n_neighbors=3)

# Entrenar el modelo con los datos
model = knn(data)

# Mostratr la tabla de datos
print("Tabla de datos:")
print(data.domain)
print(data )

# Crear una nueva instancia de datos para predecir la clase de una flor con características específicas
nueva_data = Orange.data.Table.from_list(
    data.domain,
    [[5.1, 3.5, 1.4, 0.2, None]]
)

# Realizar la predicción utilizando el modelo entrenado y obtener el índice de la clase predicha    
prediccion = model(nueva_data)
indice = int(prediccion[0])
nombre_clase = data.domain.class_var.values[indice]

# Imprimir el índice de la clase predicha y su nombre
print("Índice:", indice)
print("Clase predicha:", nombre_clase)