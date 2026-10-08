from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
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

# 3. División entrenamiento / prueba (estratificada) y escalado
X = clientes[PREDICTORES]
y = clientes["premium"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

escalador = StandardScaler()
X_train_esc = escalador.fit_transform(X_train)   # se ajusta solo con entrenamiento
X_test_esc = escalador.transform(X_test)

registrar("\n3. DIVISIÓN")
registrar(f"Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,} | "
          f"Premium en prueba: {y_test.mean():.2%}")
registrar(f"Exactitud de referencia (predecir siempre Normal): {1 - y_test.mean():.2%}")

# 4. Entrenamiento y evaluación
logistica = LogisticRegression(max_iter=1000)
logistica.fit(X_train_esc, y_train)
pred_log = logistica.predict(X_test_esc)

arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol.fit(X_train, y_train)    # el árbol no necesita escalado
pred_arbol = arbol.predict(X_test)

registrar("\n4. RESULTADOS EN PRUEBA")
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
importancia = pd.DataFrame({
    "coef_logistica": logistica.coef_[0],
    "importancia_arbol": arbol.feature_importances_,
}, index=PREDICTORES).sort_values("importancia_arbol", ascending=False)
registrar("\n5. INTERPRETACIÓN")
registrar("Coeficientes de la regresión (variables estandarizadas) e importancia en el árbol:")
registrar(importancia.round(3).to_string())
registrar("\nReglas del árbol:")
registrar(export_text(arbol, feature_names=PREDICTORES, class_names=CLASES))

# Figura 06: matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, nombre, pred in [(axes[0], "Regresión logística", pred_log),
                         (axes[1], "Árbol de decisión", pred_arbol)]:
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASES,
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{nombre}\nF1 Premium = {tabla.loc[nombre, 'f1_premium']:.3f}")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
fig.suptitle("Matrices de confusión (conjunto de prueba)")
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