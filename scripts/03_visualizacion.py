from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
MESES = ["Dic", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov"]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

df = pd.read_parquet(PROCESSED / "ventas_limpias.parquet")
productos = pd.read_parquet(PROCESSED / "productos.parquet")
df = df.merge(productos[["StockCode", "Producto"]], on="StockCode")
df["Mes"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
completo = df[df["InvoiceDate"] < "2011-12-01"]  # excluye diciembre 2011 (9 días)

def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIGURES / nombre, dpi=150)
    plt.close(fig)
    print(f"Guardado: reports/figures/{nombre}")

# Figura 01: distribución de cantidad, precio e ingreso por línea
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
variables = [("Quantity", "Cantidad (unidades)"),
             ("Price", "Precio unitario (£)"),
             ("Revenue", "Ingreso por línea (£)")]
for ax, (col, etiqueta) in zip(axes, variables):
    bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 50)
    if col == "Quantity":  # variable entera: límites enteros para evitar intervalos vacíos
        bins = np.unique(bins.astype(int))
    ax.hist(df[col], bins=bins, color="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    mediana = df[col].median()
    ax.axvline(mediana, color="red", linestyle="--", label=f"Mediana: {mediana:,.2f}")
    ax.set_xlabel(etiqueta)
    ax.set_ylabel("Líneas de factura")
    ax.legend()
fig.suptitle("Distribución por línea de factura (escala logarítmica en ambos ejes)")
guardar(fig, "01_distribuciones.png")

# Figura 02: top 10 productos por unidades y por ingreso
top_u = productos.nlargest(10, "unidades")
top_i = productos.nlargest(10, "ingreso")
en_ambos = set(top_u["StockCode"]) & set(top_i["StockCode"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, top, col, etiqueta in [(axes[0], top_u, "unidades", "Unidades vendidas"),
                               (axes[1], top_i, "ingreso", "Ingreso (£)")]:
    colores = ["darkorange" if c in en_ambos else "steelblue" for c in top["StockCode"]]
    ax.barh(top["Producto"], top[col], color=colores)
    ax.invert_yaxis()
    ax.set_xlabel(etiqueta)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_title(f"Top 10 por {col}")
fig.suptitle("Productos más vendidos (naranja: aparece en ambos rankings)")
guardar(fig, "02_top_productos.png")

# Figura 03: tendencia mensual de los 5 productos con mayor ingreso
top5 = productos.nlargest(5, "ingreso")["StockCode"]
tendencia = (completo[completo["StockCode"].isin(top5)]
             .pivot_table(index="Mes", columns="Producto", values="Revenue", aggfunc="sum")
             .fillna(0))

tendencia.index = tendencia.index.strftime("%Y-%m")

fig, ax = plt.subplots(figsize=(13, 5.5))
tendencia.plot(ax=ax, marker="o", markersize=3)
ax.set_xticks(range(0, len(tendencia), 3))
ax.set_xticklabels(tendencia.index[::3])
ax.set_xlabel("")
ax.set_ylabel("Ingreso mensual (£)")
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.set_title("Ingreso mensual de los 5 productos con mayor ingreso (dic 2009 – nov 2011)")
ax.legend(title="", fontsize=9)
guardar(fig, "03_tendencia_top_productos.png")

# Figura 04: estacionalidad, comparación de los dos años comerciales (dic-nov)
completo = completo.assign(
    AnioComercial=completo["InvoiceDate"].dt.year + (completo["InvoiceDate"].dt.month == 12),
    NumMes=completo["InvoiceDate"].dt.month)
interanual = (completo.groupby(["NumMes", "AnioComercial"])["Revenue"].sum()
              .unstack().reindex([12] + list(range(1, 12))) / 1000)

fig, ax = plt.subplots(figsize=(11, 5))
for anio in interanual.columns:
    serie = interanual[anio]
    etiqueta = (f"Dic {anio - 1} – Nov {anio} "
                f"(máximo: {MESES[serie.argmax()]}, £{serie.max():,.0f} mil)")
    ax.plot(MESES, serie, marker="o", label=etiqueta)
ax.set_ylabel("Ingreso mensual (miles de £)")
ax.set_title("Ingreso mensual: comparación entre los dos años")
ax.legend()
guardar(fig, "04_estacionalidad_interanual.png")

# Figura 05: facturas por día de la semana y hora
facturas = df.drop_duplicates("Invoice")
matriz = pd.crosstab(facturas["InvoiceDate"].dt.dayofweek, facturas["InvoiceDate"].dt.hour)
matriz = matriz.reindex(range(7), fill_value=0)
matriz.index = DIAS

fig, ax = plt.subplots(figsize=(13, 4.5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Facturas"}, ax=ax)
ax.set_xlabel("Hora")
ax.set_ylabel("")
ax.set_title("Facturas por día de la semana y hora (dic 2009 – dic 2011)")
guardar(fig, "05_facturas_dia_hora.png")