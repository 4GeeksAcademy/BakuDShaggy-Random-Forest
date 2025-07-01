from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import json

print("="*50)
print(" Proyecto : Prediccion de Diabetes con random forest")
print("="*50)
# Paso 1: Carga de datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

# Exploración inicial
print("Primeras filas:")
print(df.head())
print("\nResumen estadístico:")
print(df.describe())
print("\nValores faltantes:")
print(df.isnull().sum())

# Histogramas
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

# Correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.savefig('correlation_matrix.png')
plt.close()

# Preprocesamiento
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Dividir datos
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")
print(f"Proporción de diabetes en entrenamiento: {y_train.mean():.2f}")
print(f"Proporción de diabetes en prueba: {y_test.mean():.2f}")

# Paso 2: Modelo Árbol de Decisión
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"\n[ÁRBOL DE DECISIÓN] Precisión: {accuracy_dt:.2f}")

cm = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.savefig('confusion_matrix_dt.png')
plt.close()

# Paso 3: Modelo Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\n[MEJORES HIPERPARÁMETROS] {best_params}")

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"[RANDOM FOREST] Precisión: {accuracy_rf:.2f}")

print("\nCOMPARACIÓN DE MODELOS:")
print(f"- Árbol de Decisión: {accuracy_dt:.2f}")
print(f"- Random Forest: {accuracy_rf:.2f}")
print(f"Mejora: {(accuracy_rf - accuracy_dt)*100:.1f}%")

feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind='barh')
plt.title('Importancia de Características (Random Forest)')
plt.savefig('feature_importance.png')
plt.close()

# Paso 4: Guardar resultados
joblib.dump(best_rf, 'random_forest_model.pkl')

results = {
    'model': 'Random Forest',
    'accuracy': accuracy_rf,
    'best_params': best_params,
    'feature_importances': feature_importances.to_dict()
}

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nModelo guardado como 'random_forest_model.pkl'")
print("Resultados guardados en 'model_results.json'")