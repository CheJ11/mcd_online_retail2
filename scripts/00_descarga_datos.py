"""
00_descarga_datos.py
Descarga Online Retail II (UCI, ID 502), verifica su integridad y estructura,
y guarda una copia de trabajo en Parquet.

Salidas:
    data/raw/online_retail_II.xlsx      (original, solo lectura)
    data/interim/online_retail_ii.parquet
    reports/00_registro_descarga.txt
"""
import hashlib
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
PARQUET_PATH = INTERIM / "online_retail_ii.parquet"
LOG_PATH = REPORTS / "00_registro_descarga.txt"

# Columnas esperadas según la versión II (se verifican, no se asumen)
COLUMNAS_ESPERADAS = ["Invoice", "StockCode", "Description", "Quantity",
                      "InvoiceDate", "Price", "Customer ID", "Country"]

log = []


def registrar(msg=""):
    print(msg)
    log.append(str(msg))


def sha256(path, bloque=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


# 1. Descarga (idempotente: no se repite si el zip ya existe)
for d in (RAW, INTERIM, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print(f"Descargando {URL} ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)

registrar(f"Fecha de ejecución : {datetime.now():%Y-%m-%d %H:%M}")
registrar(f"Fuente             : {URL}")
registrar(f"Tamaño zip         : {ZIP_PATH.stat().st_size / 1e6:.1f} MB")
registrar(f"SHA-256 zip        : {sha256(ZIP_PATH)}")

# 2. Extracción: se localiza el .xlsx dentro del zip en lugar de fijar su nombre
with zipfile.ZipFile(ZIP_PATH) as z:
    registrar(f"Contenido del zip  : {z.namelist()}")
    nombre_xlsx = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
    xlsx_path = RAW / Path(nombre_xlsx).name
    if not xlsx_path.exists():
        with z.open(nombre_xlsx) as src, open(xlsx_path, "wb") as dst:
            dst.write(src.read())

# Solo lectura para proteger el original
os.chmod(xlsx_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
registrar(f"SHA-256 xlsx       : {sha256(xlsx_path)}")

# 3. Lectura de todas las hojas (openpyxl tarda unos minutos con ~1M filas)
print("Leyendo Excel (puede tardar varios minutos)...")
hojas = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl",
                      dtype={"Invoice": str, "StockCode": str})

# 4. Verificación de estructura por hoja
registrar("\n--- Hojas ---")
for nombre, df in hojas.items():
    registrar(f"{nombre!r}: {df.shape[0]:,} filas x {df.shape[1]} columnas | "
              f"{df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    faltan = set(COLUMNAS_ESPERADAS) - set(df.columns)
    sobran = set(df.columns) - set(COLUMNAS_ESPERADAS)
    if faltan or sobran:
        registrar(f"  ADVERTENCIA columnas. Faltan: {faltan} | Sobran: {sobran}")

# 5. Solapamiento entre hojas (solo se mide; se resuelve en la limpieza)
if len(hojas) == 2:
    (n1, h1), (n2, h2) = hojas.items()
    ini = max(h1["InvoiceDate"].min(), h2["InvoiceDate"].min())
    fin = min(h1["InvoiceDate"].max(), h2["InvoiceDate"].max())
    registrar("\n--- Solapamiento entre hojas ---")
    if ini <= fin:
        en_rango = lambda h: h[h["InvoiceDate"].between(ini, fin)]
        a, b = en_rango(h1), en_rango(h2)
        comunes = len(a.merge(b.drop_duplicates(), how="inner"))
        registrar(f"Rango común        : {ini} -> {fin}")
        registrar(f"Filas en rango     : {n1}={len(a):,} | {n2}={len(b):,}")
        registrar(f"Filas idénticas en ambas hojas: {comunes:,}")
    else:
        registrar("Sin solapamiento de fechas.")

# 6. Unión y tipos
df = pd.concat(hojas.values(), ignore_index=True)
df["Description"] = df["Description"].astype("string")
df["Country"] = df["Country"].astype("string")
df["Invoice"] = df["Invoice"].astype("string")
df["StockCode"] = df["StockCode"].astype("string")
df["Customer ID"] = df["Customer ID"].astype("Int64")

registrar("\n--- Dataset unido (sin limpiar) ---")
registrar(f"Filas x columnas   : {df.shape[0]:,} x {df.shape[1]}")
registrar(f"Rango de fechas    : {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
registrar("\nTipos y nulos:")
resumen = pd.DataFrame({"tipo": df.dtypes.astype(str),
                        "nulos": df.isna().sum(),
                        "nulos_%": (df.isna().mean() * 100).round(2)})
registrar(resumen.to_string())

# 7. Copia de trabajo
df.to_parquet(PARQUET_PATH, index=False)
registrar(f"\nGuardado: {PARQUET_PATH.relative_to(ROOT)} "
          f"({PARQUET_PATH.stat().st_size / 1e6:.1f} MB)")

LOG_PATH.write_text("\n".join(log), encoding="utf-8")
print(f"Registro: {LOG_PATH.relative_to(ROOT)}")