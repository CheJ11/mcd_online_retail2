# Online Retail II: Aprendizaje de máquina para el análisis de ventas

Taller 02 · Maestría en Ciencia de Datos · Universidad Yachay Tech

Clasificación, segmentación de clientes y predicción de ventas sobre el dataset *Online Retail II*, con scikit-learn.

**Objetivos del taller:**

- **Parte 0:** obtención, limpieza, análisis exploratorio y visualización de los datos.
- **Parte 1:** clasificar a los clientes en "Normal" y "Premium".
- **Parte 2:** segmentar a los clientes con K-means y Mean Shift, y comparar ambos métodos.
- **Parte 3:** predecir las ventas futuras a partir de los datos históricos.

## 1. Datos

| Característica   | Valor |
| ---------------- | ----- |
| Fuente           | [UCI Machine Learning Repository, ID 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Autor            | Daqing Chen (DOI 10.24432/C5CG6D) |
| Licencia         | CC BY 4.0 |
| Contenido        | Transacciones de un minorista británico en línea, sin tienda física, que vende artículos de regalo. Muchos clientes son mayoristas |
| Periodo          | 01/12/2009 – 09/12/2011 |
| Registros        | 1,067,371 originales · 1,044,848 tras unir las dos hojas · 997,355 tras la limpieza |
| Archivo original | `online_retail_II.xlsx` (43.5 MB), con dos hojas |

Cada fila del dataset es una **línea de factura**: un producto dentro de una compra.

## 2. Estructura del repositorio

```
├── data/
│   ├── raw/          # zip y Excel original, solo lectura
│   ├── interim/      # copia de trabajo en .parquet
│   └── processed/    # ventas limpias, tabla de productos y tabla de clientes
├── scripts/
│   ├── 00_descarga_datos.py    # descarga, revisión y unión de las hojas
│   ├── 01_limpieza.py          # normalización, criterios de limpieza y tabla de ventas válidas
│   ├── 02_eda.py               # análisis exploratorio tabular y tabla de productos
│   ├── 03_visualizacion.py     # figuras del EDA
│   ├── 04_clasificacion.py     # Parte 1: tabla de clientes, etiqueta Premium y modelos
│   ├── 05_segmentacion.py      # Parte 2: K-means y Mean Shift sobre RFM
│   └── 06_prediccion.py        # Parte 3: predicción de ventas diarias
├── reports/
│   ├── 00_registro_descarga.txt
│   ├── 01_registro_limpieza.txt
│   ├── 01_codigos_no_estandar.txt
│   ├── 02_registro_eda.txt
│   ├── 04_registro_clasificacion.txt
│   ├── 05_registro_segmentacion.txt
│   ├── 06_registro_prediccion.txt
│   └── figures/                # gráficas en .png
├── requirements.txt
└── README.md
```

Los archivos de `data/` no se suben al repositorio: se generan al ejecutar los scripts. Cada script escribe un registro en `reports/` con las cifras que se citan en este documento.

## 3. Ejecución

Entorno conda `mcd`. Los scripts se ejecutan en orden desde la raíz del proyecto:

```
conda activate mcd
pip install -r requirements.txt
python scripts/00_descarga_datos.py
python scripts/01_limpieza.py
python scripts/02_eda.py
python scripts/03_visualizacion.py
python scripts/04_clasificacion.py
python scripts/05_segmentacion.py
python scripts/06_prediccion.py
```

**Dependencias:**

- pandas y numpy;
- openpyxl, para leer el Excel;
- pyarrow, para leer y guardar Parquet;
- matplotlib y seaborn, para las gráficas;
- scikit-learn, para los modelos.

`00_descarga_datos.py` tarda varios minutos porque lee el Excel; los demás scripts tardan segundos.

## 4. Proceso

```
Excel original → copia .parquet → limpieza → ventas limpias → EDA y figuras
    (raw)          (interim)                  (processed)
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                            tabla de clientes      │        ventas diarias
                              │          │         │              │
                              ▼          ▼         │              ▼
                       clasificación  segmentación │         predicción
                         (Parte 1)     (Parte 2)   │         (Parte 3)
```

1. **Obtención** (`00`): descarga el dataset, une las dos hojas del Excel y guarda una copia de trabajo en Parquet. Parquet conserva los tipos de datos y se lee mucho más rápido que Excel.
2. **Limpieza** (`01`): aplica los criterios de limpieza y guarda la tabla de ventas válidas.
3. **EDA y figuras** (`02` y `03`): describen los datos y responden a las preguntas exploratorias del taller.
4. **Partes 1 a 3** (`04` a `06`): clasificación y segmentación sobre la tabla de clientes, y predicción sobre la serie de ventas diarias.

## 5. Parte 0.1: Obtención de datos

`00_descarga_datos.py`:

1. Descarga el `.zip` desde UCI, extrae el Excel y lo deja en solo lectura. Si los archivos ya existen, no los vuelve a descargar.
2. Lee **las dos hojas**. `Invoice` y `StockCode` se leen como texto para conservar los prefijos como `C` (cancelación) y los códigos alfanuméricos como `85123A`.
3. Revisa el solapamiento entre las hojas y lo elimina.
4. Une las hojas, convierte `Customer ID` a `Int64` (entero que admite valores nulos) y guarda el resultado en Parquet.

El registro completo está en [`reports/00_registro_descarga.txt`](reports/00_registro_descarga.txt).

| Hoja | Filas | Periodo |
| --- | --- | --- |
| Year 2009-2010 | 525,461 | 01/12/2009 – 09/12/2010 |
| Year 2010-2011 | 541,910 | 01/12/2010 – 09/12/2011 |

**Las hojas se solapan.** Entre el 01/12/2010 08:26 y el 09/12/2010, ambas hojas contienen 22,523 filas, y se verificó que cada fila de ese bloque tiene una fila idéntica en la otra hoja.

- Se eliminan de la primera hoja antes de unirlas: se deben a cómo se dividió el archivo, no a duplicados del sistema de ventas.
- Hacerlo en este paso evita mezclarlas con los duplicados exactos, que se tratan en la limpieza.

**Resultado:** 1,067,371 → 1,044,848 filas.

| Campo | Nulos | % |
| --- | --- | --- |
| Description | 4,275 | 0.41 |
| Customer ID | 235,287 | 22.52 |

**La documentación no coincide con el archivo.** La página de UCI describe las variables con los nombres de la versión I del dataset (`InvoiceNo`, `UnitPrice`, `CustomerID`), pero el archivo usa los de la versión II (`Invoice`, `Price`, `Customer ID`). Además, la documentación indica que no hay valores faltantes, y sí los hay.

## 6. Parte 0.2: Limpieza de datos

**Objetivo:** obtener una tabla de **ventas válidas**: un producto real, vendido en cantidad positiva, a precio positivo y no anulado. El registro completo está en [`reports/01_registro_limpieza.txt`](reports/01_registro_limpieza.txt).

### Normalización de códigos (no elimina filas)

Antes de aplicar los criterios se corrige el formato de `StockCode`, para que el mismo producto no aparezca con dos códigos distintos.

- **Espacios sobrantes:** 1 línea (`'47503J '`). Por el espacio, el código no cumplía el formato de producto y se habría eliminado como si no fuera un producto.
- **Mayúsculas y minúsculas:** 3,345 líneas tenían letras en minúscula.
  - 173 grupos de códigos difieren solo en mayúsculas.
  - En 163 de esos grupos la descripción es idéntica.
  - En los 10 restantes cambia solo la redacción: abreviaturas (`S/4` = `SET OF 4`), espacios (`POLKADOT` = `POLKA DOT`) o un cambio de nombre comercial. Por ejemplo, `84997A–D` aparece como "3 PIECE MINI DOTS CUTLERY SET" y como "CHILDRENS CUTLERY POLKADOT", con el mismo color y tipo de producto.
  - Todos se unifican en mayúsculas. Que sean el mismo producto se infiere de la descripción.

**Resultado:** 5,305 → 5,131 códigos distintos.

La normalización se hace antes de los criterios para que los duplicados y los pares venta-cancelación se detecten sobre códigos ya corregidos.

### Criterios

| # | Criterio | Filas eliminadas | ¿Por qué? |
| --- | --- | --- | --- |
| 1 | Duplicados exactos | 11,812 | Filas idénticas en todas las columnas inflarían las ventas |
| 2 | Ventas anuladas por una cancelación | 6,092 | La venta existió, pero se canceló completa. Si se conserva, infla el valor de compra del cliente |
| 3 | Cancelaciones (prefijo `C`) | 19,104 | No son ventas. Hay 19,165 líneas con `C`; 61 ya se habían eliminado como duplicados |
| 4 | Cantidad o precio ≤ 0 | 6,019 | Ajustes de inventario y registros sin valor comercial |
| 5 | Códigos que no son productos | 4,466 | Envíos, comisiones, ajustes, vales de regalo, pruebas y muestras |

**Resultado:** 1,044,848 → 997,355 filas. Se elimina el 4.55 % de los datos.

### Criterio 2: emparejamiento de cancelaciones

A cada cancelación se le busca la venta que anula: del mismo cliente, el mismo producto, el mismo precio y la misma cantidad, y anterior a la cancelación. Si hay varias ventas posibles, se toma la más reciente. Si dos cancelaciones apuntan a la misma venta, solo la primera se empareja.

- Se emparejaron 6,092 de las 19,104 cancelaciones (31.9 %), que suman 386,728 unidades y £612,916 de ingreso.
- Este criterio generaliza el criterio 2b del Taller 01. Los dos pedidos que allí se excluyeron a mano (80,995 y 74,215 unidades) son ahora las dos mayores ventas anuladas que detecta la regla.
- Es importante para las Partes 1 y 2. Por ejemplo, la única compra registrada del cliente 12346 fue anulada: sin este criterio, ese cliente aparecería entre los de mayor valor.

### Criterio 5: códigos que no son productos

Un producto tiene un código de 5 dígitos, a veces seguido de letras. Hay 38 códigos que no siguen ese patrón, y se revisó la descripción de cada uno ([`reports/01_codigos_no_estandar.txt`](reports/01_codigos_no_estandar.txt)).

| Decisión | Códigos |
| --- | --- |
| Envíos y comisiones | `POST`, `DOT`, `C2`, `BANK CHARGES`, `AMAZONFEE` |
| Ajustes contables | `M`, `ADJUST`, `ADJUST2`, `D` (descuento), `B` (deuda incobrable) |
| Vales de regalo | `GIFT_0001_*` (7 importes) |
| Pruebas y muestras | `TEST001`, `TEST002`, `S` |
| Caso límite: se elimina | `PADS`: la descripción corresponde a un producto, pero su precio es de £0.001 (17 líneas) |
| Se conservan, porque son productos | `DCGS*` (16 códigos) y `SP1002` |

**Facturas con prefijo `A`:** son 6 ajustes contables ("Adjust bad debt", código `B`, sin cliente). Los elimina el criterio 4 (precio negativo) o el 5 (código `B`). Ninguna queda en la tabla limpia.

### ¿Qué no se eliminó?

- **Filas sin cliente (22.74 %):** son ventas reales y se conservan para la predicción de ventas (Parte 3). La clasificación y la segmentación (Partes 1 y 2) se calculan por cliente, así que las excluyen.
- **Cancelaciones sin venta equivalente:** pueden ser devoluciones parciales o anulaciones de ventas anteriores a diciembre de 2009. Las ventas originales, si existen, se conservan.

Además, se eliminan los espacios sobrantes en `Description`, `Customer ID` se renombra a `CustomerID` y se calcula `Revenue = Quantity × Price`.

### Tabla limpia

| Medida | Valor |
| --- | --- |
| Líneas | 997,355 |
| Facturas | 39,264 |
| Productos | 4,716 |
| Clientes | 5,839 |
| Países | 43 |
| Ingreso total | £19,101,998.41 |

## 7. Parte 0.3: Análisis exploratorio

El registro completo está en [`reports/02_registro_eda.txt`](reports/02_registro_eda.txt).

### Metadata por campo

| Campo | Tipo | Nulos (%) | Valores únicos |
| --- | --- | --- | --- |
| Invoice | texto | 0 | 39,264 facturas |
| StockCode | texto | 0 | 4,716 productos |
| Description | texto | 0 | 5,325 |
| Quantity | entero | 0 | 513 |
| InvoiceDate | fecha y hora | 0 | 36,516 |
| Price | decimal | 0 | 683 |
| CustomerID | entero | 22.74 | 5,839 clientes |
| Country | texto | 0 | 43 países |
| Revenue | decimal (derivada) | 0 | 5,628 |

**Nombre de cada producto.** 597 códigos tienen más de una descripción. Se usa como nombre la descripción más frecuente de cada código, y los rankings se calculan por código. Hay 36 códigos que comparten nombre con otro, por eso hay 4,680 nombres para 4,716 códigos.

### Distribución de las variables clave

| Variable (por línea) | Mediana | Media | Percentil 99 | Máximo | Asimetría |
| --- | --- | --- | --- | --- | --- |
| Cantidad (unidades) | 4 | 10.83 | 100 | 19,152 | 106.6 |
| Precio unitario (£) | 2.10 | 3.33 | 16.95 | 1,157.15 | 41.9 |
| Ingreso (£) | 10.00 | 19.15 | 179.00 | 38,970.00 | 168.6 |

- **Colas largas a la derecha.** En las tres variables la media es mayor que la mediana, y el máximo supera más de 100 veces al percentil 99. Por eso las figuras usan escala logarítmica, y en la Parte 2 estas variables se transforman con logaritmo antes de agrupar.
- **Las cantidades grandes son compras mayoristas reales, no errores.** Por ejemplo, el cliente 13902 (Dinamarca) compró entre 9,000 y 19,000 unidades de varios productos a £0.10 la unidad.
- **Una factura típica** tiene 15 líneas y vale £299.71 (mediana); la media es de £486.50.

### Productos

- Solo 4 productos aparecen a la vez en el top 10 por unidades y en el top 10 por ingreso.
- **El ingreso está concentrado:** el 6.1 % del catálogo (290 productos) genera el 50 % del ingreso, y el 22 % (1,039 productos) genera el 80 %.

### Clientes

Se calcula solo con las líneas que tienen `CustomerID`, que suman el 86.52 % del ingreso.

| Grupo de clientes | Clientes | % del ingreso |
| --- | --- | --- |
| Top 1 % | 58 | 31.04 |
| Top 5 % | 292 | 51.13 |
| Top 10 % | 584 | 63.19 |
| Top 20 % | 1,168 | 76.73 |

- El 27.62 % de los clientes compró una sola vez.
- El cliente mediano tiene 3 facturas y un ingreso de £851. El de mayor ingreso suma £579,129.
- Esta concentración es la base para definir a los clientes "Premium" en la Parte 1.

### Países

El Reino Unido concentra el 85.46 % del ingreso. Irlanda (EIRE) es el segundo país (3.17 %) con solo 3 clientes, lo que indica que son mayoristas.

## 8. Parte 0.4: Visualización

| Figura | Pregunta | Técnica |
| --- | --- | --- |
| 01 | Distribución de cantidad, precio e ingreso | Histogramas en escala logarítmica |
| 02 | Productos más vendidos | Barras horizontales, dos rankings |
| 03 | Tendencia de los productos principales | Series mensuales |
| 04 | Patrones estacionales | Comparación de los dos años |
| 05 | Concentración por día y hora | Mapa de calor |

Diciembre de 2011 se excluye de las figuras 03 y 04 porque solo tiene 9 días de datos.

### Pregunta 1: ¿Cómo se distribuyen la cantidad, el precio y el ingreso por línea?

![Distribuciones](reports/figures/01_distribuciones.png)

**Respuesta**

- Las tres variables son aproximadamente simétricas en escala logarítmica, con cola larga a la derecha: la mayoría de las líneas son pequeñas y unas pocas son muy grandes.
- **Cantidad:** el valor más frecuente es 1 unidad y la mediana es 4. Hay picos en 12 y 24 unidades, que indican ventas por docenas.
- **Precio:** tiene picos en valores redondos, que reflejan una lista de precios fija.
- **Cómo leer la cantidad:** cada una de las primeras barras corresponde a un solo valor entero (1, 2, 3…). Son más anchas porque, en escala logarítmica, la distancia entre 1 y 2 es mayor que la distancia entre 100 y 101.

### Pregunta 2: ¿Cuáles son los productos más vendidos?

![Top productos](reports/figures/02_top_productos.png)

**Respuesta**

Los dos rankings separan dos tipos de producto:

- **De volumen:** WORLD WAR 2 GLIDERS es el primero en unidades (105,755) y no aparece en el top 10 por ingreso.
- **De valor:** REGENCY CAKESTAND 3 TIER es el primero en ingreso (£323 mil) con solo 25,825 unidades.
- **De volumen y valor:** los cuatro productos en naranja aparecen en ambos rankings. Son WHITE HANGING HEART T-LIGHT HOLDER, JUMBO BAG RED RETROSPOT, ASSORTED COLOUR BIRD ORNAMENT y SMALL POPCORN HOLDER.

### Pregunta 3: ¿Cómo evolucionan las ventas de los productos principales?

![Tendencia top productos](reports/figures/03_tendencia_top_productos.png)

**Respuesta**

- **REGENCY CAKESTAND 3 TIER** empieza a venderse el 15/03/2010 y, aun así, es el producto con más ingreso del periodo. Su máximo es de septiembre a noviembre de 2010.
- **Las series varían mucho de un mes a otro**, y no todas siguen la estacionalidad general. PARTY BUNTING, por ejemplo, tiene su máximo en mayo de 2011, un patrón compatible con un producto de verano.
- **WHITE HANGING HEART T-LIGHT HOLDER** es el más estable: se vende de forma constante durante todo el periodo.

### Pregunta 4: ¿Hay patrones estacionales? ¿Cuáles son los periodos de mayor venta?

![Estacionalidad](reports/figures/04_estacionalidad_interanual.png)

**Respuesta**

- **Sí.** En los dos años las ventas suben desde septiembre y alcanzan su máximo en noviembre (£1,415 mil y £1,439 mil). Septiembre, octubre y noviembre suman el 36.1 % y el 37.5 % del ingreso anual.
- Con dos años de datos se confirma lo que el Taller 01 no pudo confirmar con uno: el pico de fin de año se repite. Es compatible con la temporada navideña de un negocio de regalos.
- **El total anual creció un 1.90 %.** El segundo año supera al primero de mayo a septiembre y queda por debajo de enero a abril.
- **La mayor diferencia es abril de 2011 (−20.3 %).** Ese mes tuvo menos días con ventas (21 frente a 23), lo que explica parte de la caída, pero no toda.

### Pregunta 5: ¿En qué días y horas se concentran las compras?

![Facturas por día y hora](reports/figures/05_facturas_dia_hora.png)

**Respuesta**

- Entre semana, las compras se concentran de 10 a 15 h, con el máximo a las 12 h todos los días.
- El jueves es el día con más facturas (20.65 %) y el único con actividad entre las 18 y las 20 h.
- El domingo solo hay compras de 10 a 16 h.
- **La tienda no vende los sábados.** Las 30 facturas en sábado son todas del 05/12/2009, una excepción puntual.

## 9. Parte 1: Clasificación de clientes

El registro completo está en [`reports/04_registro_clasificacion.txt`](reports/04_registro_clasificacion.txt).

### Unidad de análisis

Lo que se clasifica es al cliente, así que se construye una tabla con **una fila por cliente** a partir de las líneas con `CustomerID` (5,839 clientes). La columna de la etiqueta, `premium`, se define en esa tabla. La recencia y la antigüedad se miden en días hasta el 10/12/2011, el día siguiente a la última factura.

### ¿Cómo definir "Premium"?

Se evaluaron varias alternativas:

| Criterio | Qué mide | Limitación |
| --- | --- | --- |
| **Top 20 % por ingreso (elegido)** | Valor | Un comprador con un único pedido grande cuenta como Premium |
| Umbral fijo en £ | Valor | Sin un criterio del negocio, el valor es arbitrario |
| Frecuencia de compra | Lealtad | Ignora el valor y tiene muchos empates |
| Ingreso y frecuencia combinados | Valor recurrente | Exige dos umbrales arbitrarios |
| Puntuación RFM | Comportamiento completo | Deja pocos predictores y se superpone con la Parte 2 |
| Valor en el año siguiente | Valor futuro | Mezcla el bajo valor con el abandono |

**Definición:** Premium = ingreso total ≥ percentil 80 del ingreso por cliente (£2,864.06).

**Por qué el 20 %.** La tabla de concentración del EDA muestra que el 20 % de los clientes genera el 76.73 % del ingreso, casi la regla 80/20 de Pareto. El corte separa a un grupo pequeño de clientes con una aportación muy alta. Otro porcentaje también sería defendible: los datos respaldan el 20 %, pero no lo determinan.

| | Valor |
| --- | --- |
| Clientes Premium | 1,168 (20 %) |
| Ingreso de los Premium | 76.73 % del total |
| Premium con una sola factura | 10 (0.9 %) |

### Variables

| Variable | Definición | ¿Predictor? |
| --- | --- | --- |
| `ingreso` | Suma del ingreso | **No**: define la etiqueta |
| `unidades` | Suma de las cantidades | **No**: está muy correlacionada con el ingreso |
| `frecuencia` | Número de facturas | Sí |
| `recencia` | Días desde la última compra | Sí |
| `antiguedad` | Días desde la primera compra | Sí |
| `productos_distintos` | Productos distintos comprados | Sí |
| `lineas_por_factura` | Líneas por factura, en promedio | Sí |
| `es_uk` | 1 si el cliente es del Reino Unido | Sí |

**Por qué se excluyen el ingreso y las unidades.** Si la variable que define la etiqueta fuera también un predictor, el modelo solo aprendería el umbral de £2,864. Tendría una exactitud cercana al 100 % sin aprender nada sobre el comportamiento del cliente.

**Perfil por clase (mediana):**

| | Normal | Premium |
| --- | --- | --- |
| Ingreso (£) | 596 | 5,320 |
| Frecuencia | 2 | 12 |
| Recencia (días) | 169 | 23 |
| Antigüedad (días) | 456 | 679 |
| Productos distintos | 34 | 162 |

### Modelos

- **División:** 80 % entrenamiento y 20 % prueba, estratificada para conservar la proporción 80/20 de las clases en ambos conjuntos.
- **Referencia:** predecir siempre "Normal" da un 79.97 % de exactitud sin aprender nada. Por eso la métrica principal es el **F1 de la clase Premium**, no la exactitud.
- **Regresión logística:** va en un `Pipeline` con `StandardScaler`. Así, en cada partición de la validación cruzada, el escalado se ajusta solo con los datos de entrenamiento de esa partición.
- **Árbol de decisión:** no necesita escalado.

### Optimización de hiperparámetros

Los hiperparámetros se eligieron con `GridSearchCV`: validación cruzada de 5 particiones **solo sobre el conjunto de entrenamiento**, con el F1 de Premium como métrica. El conjunto de prueba se usó una sola vez, al final.

| Modelo | Hiperparámetros probados | Elegidos |
| --- | --- | --- |
| Regresión logística | `C`: 0.01, 0.1, 1, 10 · `class_weight`: None, balanced | C = 1, balanced |
| Árbol de decisión | `max_depth`: 2–8 · `min_samples_leaf`: 1, 10, 25, 50 · `class_weight`: None, balanced | max_depth = 4, min_samples_leaf = 25, None |

| Modelo | F1 CV inicial | F1 CV optimizado | Mejora |
| --- | --- | --- | --- |
| Regresión logística | 0.772 | 0.775 | +0.002 |
| Árbol de decisión | 0.777 | 0.779 | +0.002 |

**La optimización no mejoró los modelos de forma significativa.** La mejora (+0.002) es mucho menor que la variación entre particiones (de 0.010 a 0.024), y las cinco mejores combinaciones de cada modelo difieren en menos de 0.005. La configuración inicial ya estaba cerca del óptimo: el límite del rendimiento está en la información de las variables, no en los hiperparámetros.

### Resultados

| Modelo | Exactitud | Precisión Premium | Recall Premium | F1 Premium |
| --- | --- | --- | --- | --- |
| Referencia | 0.800 | — | 0.000 | 0.000 |
| Regresión logística | 0.892 | 0.679 | 0.876 | 0.765 |
| Árbol de decisión | 0.916 | 0.781 | 0.808 | 0.794 |

![Matrices de confusión](reports/figures/06_matrices_confusion.png)

- **Ambos modelos superan claramente a la referencia.** El F1 de prueba es similar al de validación cruzada (0.765 frente a 0.775, y 0.794 frente a 0.779), así que la elección de hiperparámetros no sobreajustó.
- **El efecto de `class_weight="balanced"`.** Con pesos balanceados, la regresión logística detecta más Premium (205 de 234, frente a 171 sin pesos), a cambio de más falsas alarmas (97 Normales clasificados como Premium, frente a 33). El F1 casi no cambia, pero sí la forma de equivocarse:
  - si el objetivo es no perder a ningún Premium (por ejemplo, en una campaña de retención de bajo costo), conviene la versión balanceada;
  - si cada acción sobre un cliente es costosa, conviene la versión sin pesos, que se equivoca menos al señalar Premium.
- **El árbol tiene el mejor F1 (0.794)** y un equilibrio entre precisión y recall.

### Interpretación

![Importancia de variables](reports/figures/07_importancia_variables.png)

- **La frecuencia domina en ambos modelos:** tiene una importancia de 0.91 en el árbol y un coeficiente de 5.03 en la regresión.
  - El árbol se reduce casi a una regla: "más de 7.5 facturas → Premium".
  - Es esperable: el ingreso es aproximadamente la frecuencia por el valor medio de cada factura, así que comprar a menudo explica la mayor parte de ser Premium.
- **Ser extranjero aumenta la probabilidad de ser Premium** con la misma frecuencia (coeficiente negativo de `es_uk`). El árbol tiene una regla en el mismo sentido: "no es del Reino Unido y compró más de 104.5 productos distintos → Premium". Encaja con el EDA: los clientes extranjeros suelen ser mayoristas.
- **También aumentan la probabilidad de ser Premium:** comprar más productos distintos, tener más antigüedad y haber comprado más recientemente.
- **El árbol no usa la recencia ni la antigüedad:** ambas tienen importancia 0.
- **Varias divisiones del árbol llevan a la misma clase en ambas ramas.** No es un error: el árbol divide cuando los grupos resultantes son más homogéneos, aunque la clase mayoritaria no cambie.

### Limitaciones

- **La frecuencia explica gran parte del resultado**, y es una aproximación parcial del ingreso. El modelo identifica bien a los Premium que compran a menudo. Los errores probablemente corresponden a clientes con pocos pedidos grandes o con muchos pedidos pequeños, aunque no se ha verificado.
- **Variables correlacionadas.** La frecuencia, los productos distintos y la antigüedad están correlacionados. Los coeficientes indican la dirección del efecto, pero no cuánto aporta cada variable por sí sola.
- **Etiqueta estática.** La etiqueta describe a los clientes durante todo el periodo; no predice quién será Premium en el futuro.

## 10. Parte 2: Segmentación de clientes

El registro completo está en [`reports/05_registro_segmentacion.txt`](reports/05_registro_segmentacion.txt).

### Variables y preparación

Se usan las tres variables RFM de la tabla de clientes: **recencia** (días desde la última compra), **frecuencia** (número de facturas) e **ingreso** total. Aquí el ingreso sí se usa, porque no hay una etiqueta que predecir.

| Variable | Asimetría original | Tras log1p |
| --- | --- | --- |
| Recencia | 0.89 | −0.55 |
| Frecuencia | 11.96 | 1.01 |
| Ingreso | 26.59 | 0.25 |

1. **`log1p`** (logaritmo de 1 + x). Sin esta transformación, los pocos clientes extremos (con un ingreso de hasta £579 mil) dominarían las distancias y formarían grupos propios. Se usa `log1p` y no `log` porque la recencia puede valer 0.
2. **`StandardScaler`.** Los dos métodos miden distancias. Sin escalar, la recencia, que va de 0 a 738 días, pesaría más que las otras variables.

### K-means

**Elección de k:** se probaron de 2 a 8 grupos.

![Codo y silueta](reports/figures/08_kmeans_codo_silueta.png)

- **Silueta.** La mejor es la de k = 2 (0.439), pero dos grupos solo separan a los buenos clientes del resto. Entre las opciones con más grupos, k = 4 es un máximo local (0.364), mejor que k = 3 (0.347) y que k = 5 (0.341).
- **Codo.** La curva de inercia no tiene un codo marcado. A partir de k = 4, cada grupo adicional reduce cada vez menos la inercia.

**Segmentos con k = 4:**

| Grupo | Clientes | Recencia (días) | Frecuencia | Ingreso | % del ingreso | % Premium | Segmento |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1,978 (33.9 %) | 400 | 1 | £270 | 3.8 | 0.0 | Perdidos: una compra, hace más de un año |
| 1 | 1,232 (21.1 %) | 23 | 3 | £717 | 6.2 | 1.3 | Recientes de bajo valor |
| 2 | 1,468 (25.1 %) | 178 | 5 | £1,455 | 16.9 | 15.8 | En riesgo: valor medio y meses sin comprar |
| 3 | 1,161 (19.9 %) | 16 | 13 | £4,922 | 73.1 | 79.2 | Mejores clientes: frecuentes, recientes y de alto valor |

Los valores de recencia, frecuencia e ingreso son medianas. El grupo 3 coincide en un 79 % con los clientes Premium de la Parte 1: el agrupamiento, sin ninguna etiqueta, llega al mismo núcleo de clientes que la clasificación.

### Mean Shift

Mean Shift no recibe el número de grupos: lo determina el **ancho de banda**, que se calcula con `estimate_bandwidth` a partir de un cuantil. Un cuantil menor da un ancho de banda menor y, por tanto, más grupos.

| Cuantil | Grupos | Grupo menor | Silueta |
| --- | --- | --- | --- |
| 0.05 | 25 | 1 | 0.167 |
| 0.10 | 9 | 1 | 0.256 |
| 0.15 | 3 | 24 | 0.411 |
| **0.20** | **2** | **396** | **0.438** |
| 0.25 | 2 | 499 | 0.430 |
| 0.30 | 1 | 5,839 | — |

**Se eligió el cuantil 0.20:** tiene la mejor silueta sin grupos diminutos.

- Con 0.05 y 0.10 aparecen grupos de un solo cliente.
- Con 0.15 aparece un grupo de solo 24 clientes (0.4 %).
- Con 0.20 y 0.25 la estructura es la misma, lo que indica que no depende de un valor concreto del parámetro.

| Grupo | Clientes | Recencia (días) | Frecuencia | Ingreso | % del ingreso | % Premium |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 5,443 (93.2 %) | 119 | 3 | £757 | 46.5 | 14.2 |
| 1 | 396 (6.8 %) | 10 | 26 | £10,544 | 53.5 | 99.7 |

Mean Shift encuentra una sola zona donde se concentra la mayoría de los clientes, y un grupo de **élite**: el 6.8 % de los clientes genera más de la mitad del ingreso.

### Comparación

![Segmentos](reports/figures/09_segmentos.png)

| | K-means | Mean Shift |
| --- | --- | --- |
| Número de grupos | Lo fija el usuario (k = 4) | Lo determina el algoritmo (2) |
| Silueta | 0.364 (k = 4); 0.439 (k = 2) | 0.438 |
| Tamaño de los grupos | Parecidos (20–34 %) | Muy desiguales (93 % y 7 %) |
| Qué describe | Etapas del cliente: perdido, reciente, en riesgo, mejor | El grueso de los clientes frente a la élite |

- **Con el mismo número de grupos, la calidad es la misma:** K-means con k = 2 tiene una silueta de 0.439 y Mean Shift de 0.438. Comparar 0.438 con 0.364 no sería justo, porque la silueta suele ser mayor cuando hay menos grupos.
- **Los resultados son coherentes entre sí:** 395 de los 396 clientes de la élite de Mean Shift están en el grupo 3 de K-means. Mean Shift separa la parte más alta de ese grupo.
- **Cada método sirve para algo distinto.** K-means da segmentos con los que se puede actuar, por ejemplo reactivar a los clientes "en riesgo". Mean Shift muestra que, por la forma en que se concentran los clientes, la única separación clara es la de la élite.
- **Nota sobre la figura:** muestra solo frecuencia e ingreso. Los grupos de K-means se superponen porque la recencia, que no aparece en el gráfico, también los separa. Los grupos 1 y 2, por ejemplo, tienen frecuencias parecidas pero recencias muy distintas (23 y 178 días).

### Limitaciones

- **La silueta mide la separación geométrica de los grupos, no si son útiles para el negocio.** La elección de k = 4 combina ambos criterios.
- **Los resultados dependen de la transformación `log1p`:** sin ella, los grupos serían distintos.

## 11. Parte 3: Predicción de ventas

El registro completo está en [`reports/06_registro_prediccion.txt`](reports/06_registro_prediccion.txt).

### Qué se predice

Se predice el **ingreso total de cada día con ventas** (603 días). Se incluyen las ventas sin cliente, porque son ventas reales. El 09/12/2011 se excluye porque los datos terminan a mediodía.

El modelo usa solo **variables de calendario**. Como se conocen de antemano para cualquier fecha, permiten predecir toda una temporada antes de que empiece, que es lo que necesita un negocio de regalos para preparar su inventario.

### Variables

| Variable | Qué captura |
| --- | --- |
| `dia_semana` | El patrón semanal (el jueves es el día de más ventas; los sábados no hay ventas) |
| `mes` | La estacionalidad anual |
| `semana_anio` | La estacionalidad anual, con más detalle |
| `dia_mes` | Efectos dentro del mes |

- **Bosque aleatorio:** usa las cuatro variables como números.
- **Regresión lineal:** usa el día de la semana y el mes como variables indicadoras (*one-hot*). Si el mes entrara como número, el modelo supondría una relación en línea recta entre enero (1) y diciembre (12), y no podría reproducir la subida de la temporada.

No se usaron categorías de productos: el dataset no las incluye, y lo que se predice es la venta total diaria.

### División y modelos

- **División temporal:** entrenamiento del 01/12/2009 al 31/08/2011 (518 días) y prueba del 01/09/2011 al 08/12/2011 (85 días).
  - La prueba incluye la temporada alta, que el modelo debe predecir con lo aprendido de 2010.
  - Una división aleatoria permitiría entrenar con días posteriores a los que se predicen.
- **Referencia estacional:** cada día se predice con las ventas del mismo día de la semana, 52 semanas antes. Un modelo que no supere a "lo mismo que el año pasado" no aporta nada.
- **Regresión lineal** y **bosque aleatorio**.

### Optimización del bosque aleatorio

Se usó `GridSearchCV` con `TimeSeriesSplit` (5 particiones): cada partición de validación es posterior a su entrenamiento. Se probaron `n_estimators` (100, 300), `max_depth` (None, 5, 10) y `min_samples_leaf` (1, 5, 10).

| Configuración | MAE en validación cruzada |
| --- | --- |
| Por defecto | £8,346 |
| Optimizada (300 árboles, profundidad 5, mínimo 5 días por hoja) | £8,003 |

La mejora (−4.1 %) es menor que la variación entre particiones (unos £1,280). Las cinco mejores combinaciones exigen todas un mínimo de 5 días por hoja: limitar el detalle del árbol ayuda algo frente al ruido diario.

### Resultados

| Modelo | MAE | RMSE | R² | Error del total |
| --- | --- | --- | --- | --- |
| Referencia (hace 52 semanas) | £13,674 | £18,505 | 0.05 | −5.66 % |
| Regresión lineal | £11,141 | £16,111 | 0.28 | −8.39 % |
| Bosque aleatorio | £10,035 | £14,419 | 0.42 | −6.92 % |

![Predicción diaria](reports/figures/10_prediccion_diaria.png)

- **Predicción diaria.** El bosque aleatorio reduce el error medio un 27 % frente a la referencia y explica el 42 % de la variación diaria. Los modelos siguen el patrón semanal y la subida de la temporada, pero no los picos de días concretos, que corresponden a pedidos grandes puntuales.
- **Por qué falla la referencia:** copia los picos del año anterior en días en que no se repiten.
- **Error de prueba frente a validación cruzada.** El MAE de prueba (£10,035) es mayor que el de validación (£8,003) porque la prueba es la temporada alta: la venta media diaria es de unos £46,400, frente a £31,600 en todo el periodo. En términos relativos, el error equivale al 22 % de la venta diaria media.

![Predicción mensual](reports/figures/11_prediccion_mensual.png)

| Mes | Real | Referencia | Regresión lineal | Bosque aleatorio |
| --- | --- | --- | --- | --- |
| Sep 2011 | £1,018 mil | £868 mil | £852 mil | £863 mil |
| Oct 2011 | £1,070 mil | £1,057 mil | £1,072 mil | £1,122 mil |
| Nov 2011 | £1,439 mil | £1,445 mil | £1,412 mil | £1,328 mil |
| Dic 2011 (días 1–8) | £416 mil | £349 mil | £276 mil | £357 mil |

- **En el total de la temporada, la referencia es la que menos se equivoca (−5.66 %).** Los modelos mejoran la predicción de cada día, pero no la del total.
- **La mitad o más del error se concentra en septiembre** (del 50 % en la regresión lineal al 67 % en la referencia). Las ventas de septiembre de 2011 crecieron un 19 % frente a 2010, y ningún método podía anticiparlo: solo conocen un otoño y no tienen una variable de tendencia.
- **Octubre y noviembre se predicen bien:** la regresión lineal se desvía un 0.1 % y un 1.9 %.

**Importancia de las variables (bosque):** semana del año (0.56) y día de la semana (0.34). El mes aporta poco (0.02), porque la semana del año ya contiene esa información.

### Limitaciones

- **Un solo ciclo estacional.** El entrenamiento incluye un solo otoño, el de 2010. Con más años, los modelos podrían separar el patrón estacional de las variaciones propias de cada año.
- **No hay tendencia.** El modelo no puede anticipar crecimientos como el de septiembre de 2011. Un bosque aleatorio tampoco podría extrapolarla aunque se añadiera una variable de año, porque los árboles no predicen valores fuera del rango que vieron al entrenar.
- **Picos impredecibles.** Los días con pedidos mayoristas grandes no dependen del calendario, y son el principal origen del 58 % de la variación diaria que no se explica.

## 12. Conclusiones

1. **La calidad de los datos condiciona todo el análisis.**
   - Las dos hojas del Excel compartían 22,523 filas.
   - Había 6,092 ventas que luego se cancelaron completas.
   - 173 grupos de códigos diferían solo en mayúsculas.
   - Sin corregir esto, las ventas por cliente y por producto estarían infladas o divididas, y los modelos de las Partes 1 y 2 trabajarían con valores incorrectos.
2. **El ingreso está muy concentrado.** El 20 % de los clientes genera el 76.7 % del ingreso, y el 6.1 % de los productos genera el 50 %. Esta concentración aparece en las tres partes:
   - define a los clientes Premium (Parte 1);
   - forma el grupo de mejores clientes de K-means (19.9 % de los clientes y 73.1 % del ingreso);
   - forma la élite de Mean Shift (6.8 % de los clientes y 53.5 % del ingreso).
3. **La frecuencia de compra es la variable clave del cliente.**
   - En la clasificación concentra el 91 % de la importancia del árbol.
   - En la segmentación separa a los mejores clientes del resto.
   - Las Partes 1 y 2 llegan al mismo núcleo por caminos distintos: el 79 % del grupo 3 de K-means es Premium, y el 99.7 % de la élite de Mean Shift también lo es.
4. **La estacionalidad es fuerte y se repite.** Septiembre, octubre y noviembre concentran más de un tercio del ingreso anual, con el máximo en noviembre en ambos años. Es la variable más importante para predecir las ventas.
5. **Optimizar los hiperparámetros no cambió las conclusiones.** En la Parte 1, el F1 mejoró +0.002; en la Parte 3, el MAE bajó un 4.1 %. En ambos casos la mejora es menor que la variación entre particiones. El límite del rendimiento está en la información disponible, no en la configuración de los modelos.
6. **Un modelo debe compararse con una referencia simple.**
   - En la clasificación, una exactitud del 92 % solo tiene sentido frente al 80 % de predecir siempre "Normal".
   - En la predicción, el bosque aleatorio mejora un 27 % el error diario frente a "lo mismo que el año pasado", pero no mejora la predicción del total de la temporada.

## 13. Limitaciones generales

- **Clientes sin identificar.** El 22.74 % de las líneas no tiene cliente. Las Partes 1 y 2 describen solo a los clientes identificados, que generan el 86.5 % del ingreso.
- **Solo dos años de datos.** Hay un único ciclo estacional para entrenar la predicción, y no es posible estimar la tendencia.
- **Concentración geográfica.** El 85 % del ingreso es del Reino Unido, así que los resultados describen sobre todo a ese mercado.
- **Cancelaciones parciales.** El emparejamiento solo detecta las cancelaciones exactas (misma cantidad y precio). Las cancelaciones parciales y las que anulan ventas anteriores a diciembre de 2009 quedan en los datos.
- **No hay categorías de productos.** El dataset no las incluye, así que no se pudo analizar ni predecir por categoría.
- **Decisiones que no son únicas.** El umbral Premium (20 %), el número de grupos (k = 4) y el ancho de banda de Mean Shift se justificaron con los datos, pero otros valores también serían defendibles.