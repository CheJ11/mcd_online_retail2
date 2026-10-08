from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
LOG_PATH = ROOT / "reports" / "06_registro_prediccion.txt"

INICIO_PRUEBA = "2011-09-01"
CALENDARIO = ["dia_semana", "mes", "semana_anio", "dia_mes"]

pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")
log = []

def registrar(msg=""):
    print(msg)
    log.append(str(msg))

def metricas(real, pred):
    return {"MAE": mean_absolute_error(real, pred),
            "RMSE": np.sqrt(mean_squared_error(real, pred)),
            "R2": r2_score(real, pred),
            "error_total_%": (pred.sum() / real.sum() - 1) * 100}

# 1. Serie diaria de ventas
ventas = pd.read_parquet(PROCESSED / "ventas_limpias.parquet")
ventas["fecha"] = ventas["InvoiceDate"].dt.normalize()
diario = ventas.groupby("fecha")["Revenue"].sum().rename("ingreso").reset_index()

ultimo_dia = diario["fecha"].max()   # los datos terminan a mediodía: día incompleto
diario = diario[diario["fecha"] < ultimo_dia]

# 2. Variables de calendario
diario["dia_semana"] = diario["fecha"].dt.dayofweek
diario["mes"] = diario["fecha"].dt.month
diario["semana_anio"] = diario["fecha"].dt.isocalendar().week.astype(int)
diario["dia_mes"] = diario["fecha"].dt.day

registrar("1. SERIE DIARIA")
registrar(f"Días con ventas: {len(diario):,} | {diario['fecha'].min().date()} -> "
          f"{diario['fecha'].max().date()} (se excluye {ultimo_dia.date()}, día incompleto)")
registrar(diario["ingreso"].describe().round(2).to_string())

# 3. División temporal
entrenamiento = diario[diario["fecha"] < INICIO_PRUEBA]
prueba = diario[diario["fecha"] >= INICIO_PRUEBA]
y_train, y_test = entrenamiento["ingreso"], prueba["ingreso"]

# Bosque: variables numéricas. Regresión lineal: día de la semana y mes como indicadoras
X_train_bosque, X_test_bosque = entrenamiento[CALENDARIO], prueba[CALENDARIO]
indicadoras = pd.get_dummies(diario[["dia_semana", "mes"]].astype(str), drop_first=True)
X_train_lineal = indicadoras.loc[entrenamiento.index]
X_test_lineal = indicadoras.loc[prueba.index]

registrar("\n2. DIVISIÓN TEMPORAL")
registrar(f"Entrenamiento: {len(entrenamiento)} días | Prueba: {len(prueba)} días "
          f"({INICIO_PRUEBA} -> {prueba['fecha'].max().date()})")

# 4. Referencia estacional: mismo día de la semana, 52 semanas antes
serie = diario.set_index("fecha")["ingreso"]
hace_un_anio = serie.reindex(prueba["fecha"] - pd.Timedelta(days=364))
registrar(f"Referencia: {hace_un_anio.isna().sum()} días sin dato hace 52 semanas "
          f"(se usa la media del entrenamiento)")
pred_ref = hace_un_anio.fillna(y_train.mean()).to_numpy()

# 5. Regresión lineal
lineal = LinearRegression().fit(X_train_lineal, y_train)
pred_lineal = lineal.predict(X_test_lineal)

# 6. Bosque aleatorio: optimización con validación cruzada temporal
cv_temporal = TimeSeriesSplit(n_splits=5)
busqueda = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid={"n_estimators": [100, 300],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 5, 10]},
    cv=cv_temporal, scoring="neg_mean_absolute_error", n_jobs=-1)
busqueda.fit(X_train_bosque, y_train)
mae_inicial = -cross_val_score(RandomForestRegressor(random_state=42), X_train_bosque, y_train,
                               cv=cv_temporal, scoring="neg_mean_absolute_error").mean()

registrar("\n3. OPTIMIZACIÓN DEL BOSQUE ALEATORIO (MAE, validación cruzada temporal)")
tabla_cv = pd.DataFrame(busqueda.cv_results_)[
    ["param_n_estimators", "param_max_depth", "param_min_samples_leaf"]]
tabla_cv.columns = ["n_estimators", "max_depth", "min_samples_leaf"]
tabla_cv["MAE_cv"] = -busqueda.cv_results_["mean_test_score"]
tabla_cv["desv_cv"] = busqueda.cv_results_["std_test_score"]
registrar(tabla_cv.sort_values("MAE_cv").head(5).round(2).to_string(index=False))
registrar(f"\nMAE_cv configuración por defecto: {mae_inicial:,.2f}")
registrar(f"MAE_cv configuración optimizada: {-busqueda.best_score_:,.2f}")
registrar(f"Mejores hiperparámetros: {busqueda.best_params_}")

bosque = busqueda.best_estimator_
pred_bosque = bosque.predict(X_test_bosque)

# 7. Evaluación en prueba
predicciones = {"Referencia (hace 52 semanas)": pred_ref,
                "Regresión lineal": pred_lineal,
                "Bosque aleatorio": pred_bosque}
registrar("\n4. RESULTADOS EN PRUEBA")
resultados = pd.DataFrame({n: metricas(y_test, p) for n, p in predicciones.items()}).T
registrar(resultados.round(2).to_string())

mensual = prueba[["fecha", "ingreso"]].assign(
    referencia=pred_ref, lineal=pred_lineal, bosque=pred_bosque)
mensual = mensual.groupby(mensual["fecha"].dt.to_period("M")).sum(numeric_only=True)
registrar("\nIngreso por mes, real y predicho (£):")
registrar(mensual.round(0).to_string())

registrar("\nImportancia de variables (bosque aleatorio):")
registrar(pd.Series(bosque.feature_importances_, index=CALENDARIO)
          .sort_values(ascending=False).round(3).to_string())

# Figura 10: serie diaria en el periodo de prueba
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(prueba["fecha"], y_test, color="gray", linewidth=2, label="Real")
ax.plot(prueba["fecha"], pred_ref, color="black", linestyle=":", label="Referencia (hace 52 semanas)")
ax.plot(prueba["fecha"], pred_lineal, label="Regresión lineal")
ax.plot(prueba["fecha"], pred_bosque, label="Bosque aleatorio")
ax.set_ylabel("Ingreso diario (£)")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.set_title("Ventas diarias reales y predichas (prueba: sep – dic 2011)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "10_prediccion_diaria.png", dpi=150)
plt.close(fig)

# Figura 11: totales mensuales
fig, ax = plt.subplots(figsize=(10, 5))
etiquetas = ["Real", "Referencia", "Regresión lineal", "Bosque aleatorio"]
(mensual[["ingreso", "referencia", "lineal", "bosque"]] / 1000).set_axis(etiquetas, axis=1).plot.bar(ax=ax)
ax.set_xticklabels([str(p) for p in mensual.index], rotation=0)
ax.set_xlabel("")
ax.set_ylabel("Ingreso mensual (miles de £)")
ax.set_title("Ingreso mensual real y predicho (diciembre 2011: días 1 a 8)")
fig.tight_layout()
fig.savefig(FIGURES / "11_prediccion_mensual.png", dpi=150)
plt.close(fig)

registrar("\nGuardado: figuras 10 y 11")
LOG_PATH.write_text("\n".join(log), encoding="utf-8")