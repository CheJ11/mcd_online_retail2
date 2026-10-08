from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "data" / "processed" / "ventas_limpias.parquet"
PRODUCTOS = ROOT / "data" / "processed" / "productos.parquet"
LOG_PATH = ROOT / "reports" / "02_registro_eda.txt"

PCT = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
log = []

def registrar(msg=""):
    print(msg)
    log.append(str(msg))

def seccion(titulo):
    registrar(f"\n{titulo}")

def describir(s):
    d = s.describe(percentiles=PCT)
    d["asimetria"] = s.skew()
    d["curtosis"] = s.kurt()
    return d

df = pd.read_parquet(ENTRADA)

# Dimensión de productos: nombre canónico = descripción más frecuente del código
productos = (df.groupby("StockCode")
             .agg(Producto=("Description", lambda s: s.mode().iat[0]),
                  n_descripciones=("Description", "nunique"),
                  lineas=("StockCode", "size"),
                  unidades=("Quantity", "sum"),
                  ingreso=("Revenue", "sum"))
             .reset_index())
productos.to_parquet(PRODUCTOS, index=False)
df = df.merge(productos[["StockCode", "Producto"]], on="StockCode", how="left")

# 1. Metadata
seccion("1. METADATA")
registrar(f"Filas: {len(df):,} | Periodo: {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
meta = pd.DataFrame({"tipo": df.dtypes.astype(str),
                     "nulos_%": (df.isna().mean() * 100).round(2),
                     "unicos": df.nunique()})
registrar(meta.to_string())
registrar(f"\nCódigos con más de una descripción: "
          f"{(productos['n_descripciones'] > 1).sum():,} de {len(productos):,}")

# 2. Distribuciones
seccion("2. DISTRIBUCIONES")
registrar("Por línea de factura:")
registrar(pd.DataFrame({c: describir(df[c]) for c in ["Quantity", "Price", "Revenue"]})
          .round(2).to_string())

facturas = df.groupby("Invoice").agg(lineas=("StockCode", "size"),
                                     unidades=("Quantity", "sum"),
                                     valor=("Revenue", "sum"))
registrar("\nPor factura:")
registrar(facturas.apply(describir).round(2).to_string())

registrar("\nLíneas con mayor cantidad:")
registrar(df.nlargest(10, "Quantity")[["Invoice", "StockCode", "Producto", "Quantity",
                                       "Price", "CustomerID", "Country"]].to_string())

# 3. Productos más vendidos
seccion("3. PRODUCTOS MÁS VENDIDOS")
total_u, total_i = productos["unidades"].sum(), productos["ingreso"].sum()
cols = ["StockCode", "Producto", "unidades", "ingreso"]
top_u = productos.nlargest(10, "unidades")[cols]
top_i = productos.nlargest(10, "ingreso")[cols]
registrar("Top 10 por unidades:")
registrar(top_u.assign(**{"%_unidades": (top_u["unidades"] / total_u * 100).round(2)})
          .to_string(index=False))
registrar("\nTop 10 por ingreso:")
registrar(top_i.assign(**{"%_ingreso": (top_i["ingreso"] / total_i * 100).round(2)})
          .round(2).to_string(index=False))
registrar(f"\nProductos en ambos rankings: {len(set(top_u['StockCode']) & set(top_i['StockCode']))}")

orden = productos.sort_values("ingreso", ascending=False)["ingreso"].cumsum() / total_i
registrar(f"Productos que suman el 50% del ingreso: {(orden <= 0.5).sum() + 1:,} "
          f"({((orden <= 0.5).sum() + 1) / len(productos):.1%} del catálogo)")
registrar(f"Productos que suman el 80% del ingreso: {(orden <= 0.8).sum() + 1:,} "
          f"({((orden <= 0.8).sum() + 1) / len(productos):.1%} del catálogo)")

# 4. Temporalidad
seccion("4. TEMPORALIDAD")
df["Mes"] = df["InvoiceDate"].dt.to_period("M")
mensual = df.groupby("Mes").agg(ingreso=("Revenue", "sum"),
                                facturas=("Invoice", "nunique"),
                                dias_con_datos=("InvoiceDate", lambda s: s.dt.date.nunique()))
registrar("Ingreso mensual:")
registrar(mensual.round(2).to_string())

# Comparación interanual: años comerciales dic-nov completos
df["AnioComercial"] = df["InvoiceDate"].dt.year + (df["InvoiceDate"].dt.month == 12)
completo = df[df["InvoiceDate"] < "2011-12-01"]
interanual = (completo.groupby([completo["InvoiceDate"].dt.month, "AnioComercial"])["Revenue"]
              .sum().unstack())
interanual.columns = [f"dic{c - 1}-nov{c}" for c in interanual.columns]
interanual["variacion_%"] = (interanual.iloc[:, 1] / interanual.iloc[:, 0] - 1) * 100
orden_meses = [12] + list(range(1, 12))
interanual = interanual.reindex(orden_meses).rename_axis("mes")
registrar("\nComparación interanual (dic-nov):")
registrar(interanual.round(2).to_string())
anual = interanual.iloc[:, :2].sum()
registrar(f"Total anual: {anual.iloc[0]:,.2f} -> {anual.iloc[1]:,.2f} "
          f"({(anual.iloc[1] / anual.iloc[0] - 1) * 100:+.2f}%)")
for c in interanual.columns[:2]:
    top3 = interanual[c].nlargest(3)
    registrar(f"Meses de mayor ingreso {c}: {list(top3.index)} "
              f"({top3.sum() / interanual[c].sum():.1%} del año)")
registrar(f"Diciembre 2011 (9 días, excluido): {mensual.loc[pd.Period('2011-12'), 'ingreso']:,.2f}")

# Día de la semana y hora (por factura)
fact_t = df.drop_duplicates("Invoice")[["Invoice", "InvoiceDate"]]
dia = fact_t["InvoiceDate"].dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
dia.index = DIAS
registrar("\nFacturas por día de la semana:")
registrar(pd.DataFrame({"facturas": dia, "%": (dia / dia.sum() * 100).round(2)}).to_string())
hora = fact_t["InvoiceDate"].dt.hour.value_counts().sort_index().rename_axis(None)
registrar("\nFacturas por hora:")
registrar(pd.DataFrame({"facturas": hora, "%": (hora / hora.sum() * 100).round(2)}).T.to_string())

# 5. Clientes
seccion("5. CLIENTES (solo líneas con CustomerID)")
cli = (df.dropna(subset=["CustomerID"])
       .groupby("CustomerID")
       .agg(facturas=("Invoice", "nunique"), ingreso=("Revenue", "sum")))
registrar(f"Clientes: {len(cli):,} | Ingreso con cliente: "
          f"{cli['ingreso'].sum() / df['Revenue'].sum():.2%} del total")
registrar(f"Clientes con una sola factura: {(cli['facturas'] == 1).mean():.2%}")
registrar("\nDistribución por cliente:")
registrar(cli.apply(describir).round(2).to_string())
acum = cli["ingreso"].sort_values(ascending=False).cumsum() / cli["ingreso"].sum()
registrar("\nConcentración del ingreso (Pareto):")
for p in [0.01, 0.05, 0.10, 0.20]:
    k = max(1, int(round(p * len(cli))))
    registrar(f"  Top {p:>4.0%} de clientes ({k:,}): {acum.iloc[k - 1]:.2%} del ingreso")

# 6. Países
seccion("6. PAÍSES")
pais = df.groupby("Country").agg(lineas=("Invoice", "size"),
                                 clientes=("CustomerID", "nunique"),
                                 ingreso=("Revenue", "sum"))
pais["%_ingreso"] = pais["ingreso"] / pais["ingreso"].sum() * 100
registrar(pais.sort_values("ingreso", ascending=False).head(10).round(2).to_string())

LOG_PATH.write_text("\n".join(log), encoding="utf-8")
print(f"\nRegistro: {LOG_PATH.relative_to(ROOT)} | Productos: {PRODUCTOS.relative_to(ROOT)}")