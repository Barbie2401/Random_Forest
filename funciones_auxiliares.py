#importación de librerias base
import pandas as pd

#Librerias para las métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

#Creamos una funcion para las métricas en base a las predicciones sobre el conjunto de validación
def metricas(modelo, X_test, y_test):
	print("Test MSE: ", mean_squared_error(y_test, modelo.predict(X_test)).round(3))
	print("Test MAE: ", mean_absolute_error(y_test, modelo.predict(X_test)).round(3))
	print("Test R2: ", r2_score(y_test, modelo.predict(X_test)).round(3))