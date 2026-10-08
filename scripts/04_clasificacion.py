from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
LOG_PATH = ROOT / "reports" / "04_registro_clasificacion.txt"

PREDICTORES = ["frecuencia", "recencia", "antiguedad",
               "productos_distintos", "lineas_por_factura", "es_uk"]
CLASES = ["Normal", "Premium"]

pd.set_option("display.width", 200)
log = []

def registrar(msg=""):
    print(msg)
    log.append(str(msg))

def mejores_combinaciones(busqueda, n=5):
    """Las n combinaciones de hiperparámetros con mayor F1 en validación cruzada."""
    tabla = pd.DataFrame(busqueda.cv_results_["params"]).fillna("None")  # class_weight=None
    tabla.columns = [c.replace("logisticregression__", "") for c in tabla.columns]
    tabla["f1_cv"] = busqueda.cv_results_["mean_test_score"]
    tabla["desv_cv"] = busqueda.cv_results_["std_test_score"]
    return tabla.sort_values("f1_cv", ascending=False).head(n).round(3)

# 1. Tabla de clientes (solo líneas con CustomerID)
df = pd.read_parquet(PROCESSED / "ventas_limpias.parquet").dropna(subset=["CustomerID"])
fecha_ref = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

clientes = df.groupby("CustomerID").agg(
    ingreso=("Revenue", "sum"),
    unidades=("Quantity", "sum"),
    frecuencia=("Invoice", "nunique"),
    lineas=("Invoice", "size"),
    primera=("InvoiceDate", "min"),
    ultima=("InvoiceDate", "max"),
    productos_distintos=("StockCode", "nunique"),
    pais=("Country", "first"),
)
clientes["recencia"] = (fecha_ref - clientes["ultima"]).dt.days
clientes["antiguedad"] = (fecha_ref - clientes["primera"]).dt.days
clientes["lineas_por_factura"] = clientes["lineas"] / clientes["frecuencia"]
clientes["es_uk"] = (clientes["pais"] == "United Kingdom").astype(int)

registrar("1. TABLA DE CLIENTES")
registrar(f"Clientes: {len(clientes):,} | Fecha de referencia: {fecha_ref.date()}")
registrar(clientes[["ingreso", "unidades"] + PREDICTORES].describe().round(2).to_string())

# 2. Etiqueta: Premium = ingreso >= percentil 80
p80 = clientes["ingreso"].quantile(0.80)
clientes["premium"] = (clientes["ingreso"] >= p80).astype(int)
premium = clientes[clientes["premium"] == 1]

registrar("\n2. ETIQUETA")
registrar(f"Percentil 80 del ingreso: £{p80:,.2f}")
registrar(f"Clientes Premium: {len(premium):,} ({clientes['premium'].mean():.2%})")
registrar(f"Ingreso de los Premium: {premium['ingreso'].sum() / clientes['ingreso'].sum():.2%} del total")
registrar(f"Premium con una sola factura: {(premium['frecuencia'] == 1).sum():,}")

registrar("\nPerfil por clase (mediana):")
perfil = clientes.groupby("premium")[["ingreso"] + PREDICTORES].median()
perfil.index = CLASES
registrar(perfil.round(2).to_string())

clientes.to_parquet(PROCESSED / "clientes.parquet")

# 3. División entrenamiento / prueba (estratificada)
X = clientes[PREDICTORES]
y = clientes["premium"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

registrar("\n3. DIVISIÓN")
registrar(f"Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,} | "
          f"Premium en prueba: {y_test.mean():.2%}")
registrar(f"Exactitud de referencia (predecir siempre Normal): {1 - y_test.mean():.2%}")

# 4. Optimización de hiperparámetros: validación cruzada (5 particiones) solo en entrenamiento, la regresión va en un Pipeline para que el escalado se ajuste dentro de cada partición
busqueda_log = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    param_grid={"logisticregression__C": [0.01, 0.1, 1, 10],
                "logisticregression__class_weight": [None, "balanced"]},
    cv=5, scoring="f1")
busqueda_log.fit(X_train, y_train)

busqueda_arbol = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [2, 3, 4, 5, 6, 8],
                "min_samples_leaf": [1, 10, 25, 50],
                "class_weight": [None, "balanced"]},
    cv=5, scoring="f1")
busqueda_arbol.fit(X_train, y_train)

# Configuración inicial (versión anterior), evaluada con las mismas particiones
f1_inicial_log = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    X_train, y_train, cv=5, scoring="f1").mean()
f1_inicial_arbol = cross_val_score(
    DecisionTreeClassifier(max_depth=4, random_state=42),
    X_train, y_train, cv=5, scoring="f1").mean()

registrar("\n4. OPTIMIZACIÓN DE HIPERPARÁMETROS (F1 Premium, validación cruzada)")
registrar("Regresión logística, mejores combinaciones:")
registrar(mejores_combinaciones(busqueda_log).to_string(index=False))
registrar("\nÁrbol de decisión, mejores combinaciones:")
registrar(mejores_combinaciones(busqueda_arbol).to_string(index=False))

comparacion = pd.DataFrame({
    "f1_cv_inicial": [f1_inicial_log, f1_inicial_arbol],
    "f1_cv_optimizado": [busqueda_log.best_score_, busqueda_arbol.best_score_],
}, index=["Regresión logística", "Árbol de decisión"])
comparacion["mejora"] = comparacion["f1_cv_optimizado"] - comparacion["f1_cv_inicial"]
registrar("\nConfiguración inicial (C=1; árbol max_depth=4) frente a la optimizada:")
registrar(comparacion.round(3).to_string())
registrar(f"Mejor regresión: {str(busqueda_log.best_params_).replace('logisticregression__', '')}")
registrar(f"Mejor árbol: {busqueda_arbol.best_params_}")

# Evaluación final: el conjunto de prueba se usa una sola vez
mejor_log = busqueda_log.best_estimator_
mejor_arbol = busqueda_arbol.best_estimator_
pred_log = mejor_log.predict(X_test)
pred_arbol = mejor_arbol.predict(X_test)

registrar("\nRESULTADOS EN PRUEBA (modelos optimizados)")
resumen = {}
for nombre, pred in [("Regresión logística", pred_log), ("Árbol de decisión", pred_arbol)]:
    registrar(f"\n{nombre}")
    registrar(classification_report(y_test, pred, target_names=CLASES))
    resumen[nombre] = [accuracy_score(y_test, pred), precision_score(y_test, pred),
                       recall_score(y_test, pred), f1_score(y_test, pred)]

tabla = pd.DataFrame(resumen, index=["exactitud", "precision_premium",
                                     "recall_premium", "f1_premium"]).T
registrar("Resumen:")
registrar(tabla.round(3).to_string())

# 5. Interpretación
logistica = mejor_log.named_steps["logisticregression"]
importancia = pd.DataFrame({
    "coef_logistica": logistica.coef_[0],
    "importancia_arbol": mejor_arbol.feature_importances_,
}, index=PREDICTORES).sort_values("importancia_arbol", ascending=False)
registrar("\n5. INTERPRETACIÓN (modelos optimizados)")
registrar("Coeficientes de la regresión (variables estandarizadas) e importancia en el árbol:")
registrar(importancia.round(3).to_string())
registrar("\nReglas del árbol (se muestran hasta 4 niveles):")
registrar(export_text(mejor_arbol, feature_names=PREDICTORES, class_names=CLASES, max_depth=4))

# Figura 06: matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, nombre, pred in [(axes[0], "Regresión logística", pred_log),
                         (axes[1], "Árbol de decisión", pred_arbol)]:
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASES,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{nombre}\nF1 Premium = {tabla.loc[nombre, 'f1_premium']:.3f}")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
fig.suptitle("Matrices de confusión (conjunto de prueba, modelos optimizados)")
fig.tight_layout()
fig.savefig(FIGURES / "06_matrices_confusion.png", dpi=150)
plt.close(fig)

# Figura 07: importancia de variables
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
coef = importancia["coef_logistica"].sort_values()
axes[0].barh(coef.index, coef, color="steelblue")
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title("Regresión logística: coeficientes\n(positivo aumenta la probabilidad de Premium)")
imp = importancia["importancia_arbol"].sort_values()
axes[1].barh(imp.index, imp, color="steelblue")
axes[1].set_title("Árbol de decisión: importancia de variables")
fig.tight_layout()
fig.savefig(FIGURES / "07_importancia_variables.png", dpi=150)
plt.close(fig)

registrar("\nGuardado: data/processed/clientes.parquet, figuras 06 y 07")
LOG_PATH.write_text("\n".join(log), encoding="utf-8")