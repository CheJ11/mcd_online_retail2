from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENTRADA = ROOT / "data" / "interim" / "online_retail_ii.parquet"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SALIDA = PROCESSED / "ventas_limpias.parquet"
LOG_PATH = REPORTS / "01_registro_limpieza.txt"
CODIGOS_PATH = REPORTS / "01_codigos_no_estandar.txt"
MAYUS_PATH = REPORTS / "01_codigos_mayusculas.txt"

PATRON_PRODUCTO = r"^\d{5}[A-Za-z]*$"   # 5 dígitos + sufijo opcional de letras
EXCEPCIONES_PRODUCTO = ("DCGS", "SP1002")  # prefijos/códigos verificados como productos

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
log, criterios = [], []


def registrar(msg=""):
    print(msg)
    log.append(str(msg))


def aplicar(df, mascara_eliminar, nombre):
    n = int(mascara_eliminar.sum())
    criterios.append((nombre, n))
    registrar(f"{nombre:<45} {n:>9,}")
    return df.loc[~mascara_eliminar]


df = pd.read_parquet(ENTRADA).rename(columns={"Customer ID": "CustomerID"})
n0 = len(df)
facturas_a = df[df["Invoice"].str.startswith("A")].copy()   # diagnóstico
registrar(f"Filas iniciales: {n0:,}\n")

# Diagnóstico: prefijos no numéricos en Invoice
pref = df["Invoice"].str.extract(r"^(\D+)")[0].fillna("(numérico)")
registrar("Prefijos de Invoice:")
registrar(pref.value_counts().rename_axis(None).to_string() + "\n")

# 0. Normalización de StockCode (corrige formato, no elimina filas)
registrar("--- 0. Normalización de StockCode ---")
original = df["StockCode"]
df["StockCode"] = original.str.strip()
con_espacios = original[original != df["StockCode"]]
registrar(f"Líneas con espacios sobrantes : {len(con_espacios):,}")
registrar(f"Códigos afectados (repr)      : {[repr(c) for c in con_espacios.unique()[:20]]}")


def moda(s):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else None


# Diagnóstico previo a pasar a mayúsculas: evidencia de la decisión
cod = (df.groupby("StockCode")
       .agg(lineas=("StockCode", "size"), descripcion=("Description", moda)))
cod["clave"] = cod.index.str.upper()
multi = cod[cod.groupby("clave")["clave"].transform("size") > 1].copy()
multi["misma_descripcion"] = (multi.groupby("clave")["descripcion"]
                              .transform(lambda s: s.str.strip().str.upper().nunique() == 1))
multi = multi.sort_values(["clave", "lineas"], ascending=[True, False])
MAYUS_PATH.write_text(multi.to_string(), encoding="utf-8")
claves = multi.groupby("clave")["misma_descripcion"].first()
registrar(f"Grupos que difieren solo en mayúsculas: {len(claves)} "
          f"(misma descripción: {int(claves.sum())}, distinta: {int((~claves).sum())})")

n_cod = df["StockCode"].nunique()
mayus = df["StockCode"].str.upper()
registrar(f"Líneas con código en minúscula: {int((mayus != df['StockCode']).sum()):,}")
df["StockCode"] = mayus
registrar(f"Códigos distintos: {n_cod:,} -> {df['StockCode'].nunique():,}\n")

registrar(f"{'Criterio':<45} {'Eliminadas':>9}")

# 1. Duplicados exactos
df = aplicar(df, df.duplicated(keep="first"), "1. Duplicados exactos")

# 2. Ventas anuladas por una cancelación exacta
es_canc = df["Invoice"].str.startswith("C")
clave = ["CustomerID", "StockCode", "Price", "q"]
c = (df[es_canc & df["CustomerID"].notna()]
     .assign(q=lambda d: -d["Quantity"])[clave + ["InvoiceDate"]]
     .reset_index())
v = (df[~es_canc & (df["Quantity"] > 0) & df["CustomerID"].notna()]
     .assign(q=lambda d: d["Quantity"])[clave + ["InvoiceDate"]]
     .reset_index())
cand = c.merge(v, on=clave, suffixes=("_c", "_v"))
cand = cand[cand["InvoiceDate_v"] <= cand["InvoiceDate_c"]]
# Emparejamiento uno a uno: cada cancelación con la venta previa más reciente
cand = cand.sort_values(["InvoiceDate_c", "index_c", "InvoiceDate_v"],
                        ascending=[True, True, False])
usadas_c, usadas_v = set(), set()
for ic, iv in zip(cand["index_c"], cand["index_v"]):
    if ic not in usadas_c and iv not in usadas_v:
        usadas_c.add(ic)
        usadas_v.add(iv)

anuladas = df.loc[list(usadas_v)]
df = aplicar(df, df.index.isin(usadas_v), "2. Ventas anuladas por cancelación exacta")

# 3. Cancelaciones (todas las líneas con prefijo C)
df = aplicar(df, df["Invoice"].str.startswith("C"), "3. Cancelaciones (prefijo C)")

# 4. Cantidad o precio no positivos
df = aplicar(df, (df["Quantity"] <= 0) | (df["Price"] <= 0), "4. Cantidad o precio <= 0")

# 5. Códigos que no son productos
estandar = df["StockCode"].str.match(PATRON_PRODUCTO)
excepcion = df["StockCode"].str.upper().str.startswith(EXCEPCIONES_PRODUCTO)
no_estandar = df[~estandar]
tabla = (no_estandar.groupby("StockCode")
         .agg(lineas=("StockCode", "size"),
              descripcion=("Description", lambda s: s.mode().iat[0] if s.notna().any() else None),
              precio_mediano=("Price", "median"))
         .sort_values("lineas", ascending=False))
tabla["decision"] = ["conserva" if str(k).upper().startswith(EXCEPCIONES_PRODUCTO)
                     else "elimina" for k in tabla.index]
CODIGOS_PATH.write_text(tabla.to_string(), encoding="utf-8")
df = aplicar(df, ~estandar & ~excepcion, "5. Códigos que no son productos")

# Ajustes de formato y variable derivada
df["Description"] = df["Description"].str.strip()
df["Revenue"] = df["Quantity"] * df["Price"]

# Resumen
n1 = len(df)
registrar(f"\nFilas finales: {n1:,}  ({(n0 - n1) / n0:.2%} eliminado)")

registrar("\n--- Detalle criterio 2 ---")
registrar(f"Pares venta-cancelación: {len(usadas_v):,}")
registrar(f"Unidades anuladas      : {int(anuladas['Quantity'].sum()):,}")
registrar(f"Ingreso anulado (£)    : {(anuladas['Quantity'] * anuladas['Price']).sum():,.2f}")
registrar("Mayores ventas anuladas:")
registrar(anuladas.nlargest(5, "Quantity")[["Invoice", "StockCode", "Description",
                                            "Quantity", "InvoiceDate", "CustomerID"]].to_string())

registrar("\n--- Códigos no estándar ---")
registrar(f"Códigos distintos: {len(tabla)} | conservados: {(tabla['decision'] == 'conserva').sum()} "
          f"| detalle en {CODIGOS_PATH.relative_to(ROOT)}")

# Códigos que solo difieren en mayúsculas/minúsculas (diagnóstico)
# Diagnóstico: facturas con prefijo A
registrar("\n--- Facturas con prefijo A ---")
registrar(facturas_a[["Invoice", "StockCode", "Description", "Quantity",
                      "Price", "CustomerID"]].to_string())
registrar(f"Quedan en la tabla limpia: {int(df['Invoice'].str.startswith('A').sum())}")

registrar("\n--- Tabla limpia ---")
registrar(f"Facturas  : {df['Invoice'].nunique():,}")
registrar(f"Productos : {df['StockCode'].nunique():,}")
registrar(f"Clientes  : {df['CustomerID'].nunique():,}")
registrar(f"Países    : {df['Country'].nunique():,}")
registrar(f"Líneas sin cliente: {df['CustomerID'].isna().mean():.2%}")
registrar(f"Description nula  : {df['Description'].isna().sum():,}")
registrar(f"Ingreso total (£) : {df['Revenue'].sum():,.2f}")
registrar(f"Periodo   : {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")

df.reset_index(drop=True).to_parquet(SALIDA, index=False)
registrar(f"\nGuardado: {SALIDA.relative_to(ROOT)}")
LOG_PATH.write_text("\n".join(log), encoding="utf-8")