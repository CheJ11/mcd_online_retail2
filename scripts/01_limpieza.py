from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "data" / "interim" / "online_retail_ii.parquet"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SALIDA = PROCESSED / "ventas_limpias.parquet"
LOG_PATH = REPORTS / "01_registro_limpieza.txt"
CODIGOS_PATH = REPORTS / "01_codigos_no_estandar.txt"

PATRON_PRODUCTO = r"^\d{5}[A-Z]*$"         # 5 dígitos + letras opcionales
EXCEPCIONES = ("DCGS", "SP1002")            # códigos revisados que sí son productos

PROCESSED.mkdir(parents=True, exist_ok=True)
log = []

def registrar(msg=""):
    print(msg)
    log.append(str(msg))

def eliminar(df, mascara, nombre):
    """Elimina las filas marcadas y registra cuántas fueron."""
    registrar(f"{nombre:<45} {mascara.sum():>9,}")
    return df[~mascara]

df = pd.read_parquet(ENTRADA).rename(columns={"Customer ID": "CustomerID"})
n_inicial = len(df)
registrar(f"Filas iniciales: {n_inicial:,}")

# 1. Tipos de factura según su prefijo
registrar("\n1. PREFIJOS DE INVOICE")
prefijo = df["Invoice"].str.extract(r"^(\D*)")[0].replace("", "(numérico)")
registrar(prefijo.value_counts().rename_axis(None).to_string())

# 2. Normalización de StockCode: espacios y mayúsculas (no elimina filas)
registrar("\n2. NORMALIZACIÓN DE STOCKCODE")
codigos_antes = df["StockCode"].nunique()
codigo_limpio = df["StockCode"].str.strip().str.upper()
registrar(f"Líneas con espacios sobrantes: {(df['StockCode'] != df['StockCode'].str.strip()).sum():,}")
registrar(f"Líneas con letras en minúscula: {(df['StockCode'].str.strip() != codigo_limpio).sum():,}")
df["StockCode"] = codigo_limpio
registrar(f"Códigos distintos: {codigos_antes:,} -> {df['StockCode'].nunique():,}")

# 3. Criterios de limpieza
registrar("\n3. CRITERIOS")
registrar(f"{'Criterio':<45} {'Eliminadas':>9}")

# 3.1 Duplicados exactos
df = eliminar(df, df.duplicated(), "1. Duplicados exactos")

# 3.2 Ventas anuladas: venta y cancelación del mismo cliente, producto, precio y cantidad
es_cancelacion = df["Invoice"].str.startswith("C")
claves = ["CustomerID", "StockCode", "Price", "Cantidad"]
cancelaciones = (df[es_cancelacion & df["CustomerID"].notna()]
                 .assign(Cantidad=lambda d: -d["Quantity"])
                 .reset_index())
ventas = (df[~es_cancelacion & df["CustomerID"].notna()]
          .assign(Cantidad=lambda d: d["Quantity"])
          .reset_index())
pares = cancelaciones.merge(ventas, on=claves, suffixes=("_canc", "_venta"))
pares = pares[pares["InvoiceDate_venta"] <= pares["InvoiceDate_canc"]]
# Cada cancelación anula su venta previa más reciente, y cada venta se anula una sola vez
pares = (pares.sort_values("InvoiceDate_venta")
         .drop_duplicates("index_canc", keep="last")
         .drop_duplicates("index_venta"))
anuladas = df.loc[pares["index_venta"]]
df = eliminar(df, df.index.isin(pares["index_venta"]), "2. Ventas anuladas por una cancelación")

# 3.3 Cancelaciones
df = eliminar(df, df["Invoice"].str.startswith("C"), "3. Cancelaciones (prefijo C)")

# 3.4 Cantidad o precio no positivos
df = eliminar(df, (df["Quantity"] <= 0) | (df["Price"] <= 0), "4. Cantidad o precio <= 0")

# 3.5 Códigos que no son productos
no_estandar = ~df["StockCode"].str.match(PATRON_PRODUCTO)
es_excepcion = df["StockCode"].str.startswith(EXCEPCIONES)
tabla = (df[no_estandar]
         .groupby("StockCode")
         .agg(lineas=("Invoice", "size"),
              descripcion=("Description", "first"),
              precio_mediano=("Price", "median"))
         .sort_values("lineas", ascending=False))
tabla["decision"] = tabla.index.str.startswith(EXCEPCIONES)
tabla["decision"] = tabla["decision"].map({True: "conserva", False: "elimina"})
CODIGOS_PATH.write_text(tabla.to_string(), encoding="utf-8")
df = eliminar(df, no_estandar & ~es_excepcion, "5. Códigos que no son productos")

registrar(f"\nFilas finales: {len(df):,} ({(n_inicial - len(df)) / n_inicial:.2%} eliminado)")

# 4. Detalle de las ventas anuladas
registrar("\n4. VENTAS ANULADAS (CRITERIO 2)")
registrar(f"Unidades: {anuladas['Quantity'].sum():,} | "
          f"Ingreso: £{(anuladas['Quantity'] * anuladas['Price']).sum():,.2f}")
registrar(anuladas.nlargest(5, "Quantity")[["Invoice", "StockCode", "Description",
                                            "Quantity", "InvoiceDate", "CustomerID"]].to_string())

# 5. Ajustes finales y resumen
df["Description"] = df["Description"].str.strip()
df["Revenue"] = df["Quantity"] * df["Price"]

registrar("\n5. TABLA LIMPIA")
registrar(f"Facturas: {df['Invoice'].nunique():,}")
registrar(f"Productos: {df['StockCode'].nunique():,}")
registrar(f"Clientes: {df['CustomerID'].nunique():,}")
registrar(f"Países: {df['Country'].nunique():,}")
registrar(f"Líneas sin cliente: {df['CustomerID'].isna().mean():.2%}")
registrar(f"Ingreso total: £{df['Revenue'].sum():,.2f}")

df.reset_index(drop=True).to_parquet(SALIDA, index=False)
registrar(f"\nGuardado: {SALIDA.relative_to(ROOT)}")
LOG_PATH.write_text("\n".join(log), encoding="utf-8")