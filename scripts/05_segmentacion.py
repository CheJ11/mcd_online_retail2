from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
LOG_PATH = ROOT / "reports" / "05_registro_segmentacion.txt"

RFM = ["recencia", "frecuencia", "ingreso"]
K_ELEGIDO = 4         # se fija tras revisar el codo y la silueta (figura 08)
CUANTIL_ANCHO = 0.2   # se fija tras revisar el barrido de cuantiles (sección 3)

pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")
log = []

def registrar(msg=""):
    print(msg)
    log.append(str(msg))

def ordenar_por_ingreso(etiquetas):
    """Renumera los grupos de 0 en adelante, de menor a mayor ingreso mediano."""
    medianas = clientes.groupby(etiquetas)["ingreso"].median().sort_values()
    nuevo = pd.Series(range(len(medianas)), index=medianas.index)
    return etiquetas.map(nuevo)

def perfil(columna):
    """Tamaño y mediana de R, F y M por grupo, en escala original."""
    p = clientes.groupby(columna).agg(
        clientes=("ingreso", "size"),
        recencia=("recencia", "median"),
        frecuencia=("frecuencia", "median"),
        ingreso=("ingreso", "median"),
        ingreso_total=("ingreso", "sum"),
        premium=("premium", "mean"),
    )
    p["%_clientes"] = p["clientes"] / len(clientes) * 100
    p["%_ingreso"] = p["ingreso_total"] / clientes["ingreso"].sum() * 100
    p["%_premium"] = p["premium"] * 100
    return p.drop(columns=["ingreso_total", "premium"]).round(1)

# 1. Datos: log1p por la asimetría y estandarización por las escalas distintas
clientes = pd.read_parquet(PROCESSED / "clientes.parquet")
X = StandardScaler().fit_transform(np.log1p(clientes[RFM]))

registrar("1. DATOS")
registrar(f"Clientes: {len(clientes):,} | Variables: {RFM}")
registrar("Asimetría antes y después de log1p:")
registrar(pd.DataFrame({"original": clientes[RFM].skew(),
                        "log1p": np.log1p(clientes[RFM]).skew()}).round(2).to_string())

# 2. K-means: elección de k
registrar("\n2. K-MEANS: ELECCIÓN DE k")
resultados_k = []
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    resultados_k.append({"k": k, "inercia": modelo.inertia_,
                         "silueta": silhouette_score(X, modelo.labels_)})
resultados_k = pd.DataFrame(resultados_k).set_index("k")
registrar(resultados_k.round(3).to_string())

kmeans = KMeans(n_clusters=K_ELEGIDO, n_init=10, random_state=42).fit(X)
clientes["grupo_kmeans"] = ordenar_por_ingreso(pd.Series(kmeans.labels_, index=clientes.index))
sil_kmeans = silhouette_score(X, kmeans.labels_)
registrar(f"\nk elegido: {K_ELEGIDO} | Silueta: {sil_kmeans:.3f}")
registrar(perfil("grupo_kmeans").to_string())

# 3. Mean Shift: el ancho de banda depende del cuantil; el número de grupos lo decide el algoritmo
# Menor cuantil -> ancho de banda menor -> más grupos
registrar("\n3. MEAN SHIFT: ELECCIÓN DEL CUANTIL")
resultados_q = []
for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    ancho_q = estimate_bandwidth(X, quantile=q, random_state=42)
    etiquetas_q = MeanShift(bandwidth=ancho_q, bin_seeding=True).fit(X).labels_
    tamanos = pd.Series(etiquetas_q).value_counts()
    resultados_q.append({
        "cuantil": q,
        "ancho": ancho_q,
        "grupos": len(tamanos),
        "grupo_menor": tamanos.min(),
        "silueta": silhouette_score(X, etiquetas_q) if len(tamanos) > 1 else float("nan"),
    })
resultados_q = pd.DataFrame(resultados_q).set_index("cuantil")
registrar(resultados_q.round(3).to_string())

ancho = estimate_bandwidth(X, quantile=CUANTIL_ANCHO, random_state=42)
meanshift = MeanShift(bandwidth=ancho, bin_seeding=True).fit(X)
n_grupos = len(np.unique(meanshift.labels_))
clientes["grupo_meanshift"] = ordenar_por_ingreso(pd.Series(meanshift.labels_, index=clientes.index))
registrar(f"Cuantil: {CUANTIL_ANCHO} | Ancho de banda: {ancho:.3f} | Grupos encontrados: {n_grupos}")
sil_meanshift = silhouette_score(X, meanshift.labels_) if n_grupos > 1 else float("nan")
registrar(f"Silueta: {sil_meanshift:.3f}")
registrar(perfil("grupo_meanshift").to_string())

# 4. Comparación
registrar("\n4. COMPARACIÓN")
registrar(pd.DataFrame({"grupos": [K_ELEGIDO, n_grupos],
                        "silueta": [sil_kmeans, sil_meanshift]},
                       index=["K-means", "Mean Shift"]).round(3).to_string())
if n_grupos in resultados_k.index:   # la silueta depende del número de grupos
    registrar(f"K-means con el mismo número de grupos que Mean Shift (k = {n_grupos}): "
              f"silueta {resultados_k.loc[n_grupos, 'silueta']:.3f}")
registrar("\nTabla cruzada (filas: K-means, columnas: Mean Shift):")
registrar(pd.crosstab(clientes["grupo_kmeans"], clientes["grupo_meanshift"]).to_string())

# Figura 08: codo y silueta de K-means
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(resultados_k.index, resultados_k["inercia"], marker="o")
axes[0].set_title("Método del codo")
axes[0].set_xlabel("Número de grupos (k)")
axes[0].set_ylabel("Inercia")
axes[1].plot(resultados_k.index, resultados_k["silueta"], marker="o", color="darkorange")
axes[1].set_title("Coeficiente de silueta")
axes[1].set_xlabel("Número de grupos (k)")
axes[1].set_ylabel("Silueta")
for ax in axes:
    ax.axvline(K_ELEGIDO, color="gray", linestyle="--")
fig.suptitle(f"K-means: elección de k (línea punteada: k = {K_ELEGIDO})")
fig.tight_layout()
fig.savefig(FIGURES / "08_kmeans_codo_silueta.png", dpi=150)
plt.close(fig)

# Figura 09: grupos de cada método sobre frecuencia e ingreso
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for ax, col, titulo in [(axes[0], "grupo_kmeans", f"K-means (k = {K_ELEGIDO})"),
                        (axes[1], "grupo_meanshift", f"Mean Shift ({n_grupos} grupos)")]:
    sns.scatterplot(data=clientes, x="frecuencia", y="ingreso", hue=col,
                    palette="tab10", s=12, alpha=0.6, linewidth=0, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frecuencia (facturas)")
    ax.set_ylabel("Ingreso total (£)")
    ax.set_title(titulo)
    ax.legend(title="Grupo", markerscale=2)
fig.suptitle("Segmentos de clientes (escala logarítmica; grupos numerados de menor a mayor ingreso)")
fig.tight_layout()
fig.savefig(FIGURES / "09_segmentos.png", dpi=150)
plt.close(fig)

registrar("\nGuardado: figuras 08 y 09")
LOG_PATH.write_text("\n".join(log), encoding="utf-8")