import os
import stat
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
ZIP_PATH = RAW / "online_retail_ii.zip"
XLSX_PATH = RAW / "online_retail_II.xlsx"
PARQUET_PATH = INTERIM / "online_retail_ii.parquet"
LOG_PATH = REPORTS / "00_registro_descarga.txt"

for carpeta in (RAW, INTERIM, REPORTS):
    carpeta.mkdir(parents=True, exist_ok=True)

log = []

def registrar(msg=""):
    print(msg)
    log.append(str(msg))

# 1. Descarga y extracción (no se repiten si los archivos ya existen)
if not ZIP_PATH.exists():
    print("Descargando dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
if not XLSX_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW)
os.chmod(XLSX_PATH, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)  # solo lectura

registrar("1. DESCARGA")
registrar(f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M}")
registrar(f"Fuente: {URL}")
registrar(f"Tamaño del zip: {ZIP_PATH.stat().st_size / 1e6:.1f} MB")

# 2. Lectura de las dos hojas
# Invoice y StockCode se leen como texto para conservar prefijos como 'C' y códigos como '85123A'
print("Leyendo Excel (puede tardar varios minutos)...")
hojas = pd.read_excel(XLSX_PATH, sheet_name=None, dtype={"Invoice": str, "StockCode": str})

registrar("\n2. HOJAS")
for nombre, h in hojas.items():
    registrar(f"{nombre}: {len(h):,} filas | {h['InvoiceDate'].min()} -> {h['InvoiceDate'].max()}")
    registrar(f"  Columnas: {list(h.columns)}")

# 3. Solapamiento: la segunda hoja empieza antes de que termine la primera
h1, h2 = hojas["Year 2009-2010"], hojas["Year 2010-2011"]
inicio_h2 = h2["InvoiceDate"].min()
fin_h1 = h1["InvoiceDate"].max()
solape_h1 = h1[h1["InvoiceDate"] >= inicio_h2]
solape_h2 = h2[h2["InvoiceDate"] <= fin_h1]
iguales = len(solape_h1.merge(solape_h2.drop_duplicates()))

registrar("\n3. SOLAPAMIENTO ENTRE HOJAS")
registrar(f"Rango común: {inicio_h2} -> {fin_h1}")
registrar(f"Filas en el rango: hoja 1 = {len(solape_h1):,} | hoja 2 = {len(solape_h2):,}")
registrar(f"Filas de la hoja 1 con una fila idéntica en la hoja 2: {iguales:,}")

# Solo se elimina si el bloque está repetido completo
if not (iguales == len(solape_h1) == len(solape_h2)):
    raise ValueError("El solapamiento no es idéntico: revisar antes de eliminar.")
h1 = h1[h1["InvoiceDate"] < inicio_h2]
registrar(f"Filas eliminadas de la hoja 1: {len(solape_h1):,}")

# 4. Unión y tipos
df = pd.concat([h1, h2], ignore_index=True)
for col in ["Invoice", "StockCode", "Description", "Country"]:
    df[col] = df[col].astype("string")
df["Customer ID"] = df["Customer ID"].astype("Int64")  # entero que admite nulos

registrar("\n4. DATASET UNIDO (SIN LIMPIAR)")
registrar(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
registrar(f"Periodo: {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
resumen = pd.DataFrame({"tipo": df.dtypes.astype(str),
                        "nulos": df.isna().sum(),
                        "nulos_%": (df.isna().mean() * 100).round(2)})
registrar(resumen.to_string())

# 5. Copia de trabajo en Parquet
df.to_parquet(PARQUET_PATH, index=False)
registrar(f"\nGuardado: {PARQUET_PATH.relative_to(ROOT)} ({PARQUET_PATH.stat().st_size / 1e6:.1f} MB)")
LOG_PATH.write_text("\n".join(log), encoding="utf-8")