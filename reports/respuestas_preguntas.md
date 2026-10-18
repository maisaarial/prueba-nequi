# Respuestas a las preguntas de criterio

Todas las cifras provienen de `notebooks/01_modelo_pd_thin_file.ipynb`.

## 1. Construcción del default y sensibilidad al umbral
**Definición:** default = 1 si el crédito alcanza mora ≥ 60 días en alguno de los 3 primeros
cortes (MOB 1–3).

Al revisar el panel encontré que cada crédito tiene entre 3 y 6 cortes **independientemente de
su cosecha**. Con "peor mora en todo el historial", la etiqueta dependería del tiempo observado:
el 90+ es 7,4 % con 3 cortes y 13,2 % con 6. Por eso fijé una ventana igual para todos. MOB 1–3
es la única completa para el 100 % de la muestra. Elegí 60+ porque, con la mora subiendo 30 días
por corte, el 90+ en MOB 1–3 sólo ocurre si no se paga ninguna cuota (5,1 %), mientras que el
89,6 % de los que están en 60 al MOB 3 (y el 100 % de los que están en 90) terminan en 90+.

| Ventana | 30+ | 60+ | 90+ | % población |
|---|---|---|---|---|
| **MOB 1–3** | 28,5 % | **10,6 %** | 5,1 % | 100 % |
| MOB 1–4 | 34,3 % | 13,2 % | 8,1 % | 75 % |
| MOB 1–6 | 43,9 % | 16,7 % | 13,2 % | 24 % |
| Todo el historial | 37,1 % | 15,6 % | 10,4 % | 100 % (sesgado) |

30+ casi triplicaría la tasa (incluye mora técnica); 90+ la reduciría a la mitad o forzaría a
perder entre 25 % y 75 % de la muestra. Limitación: ≈ 37 % de los 90+ observados con ventana
larga caen después del MOB 3; se debe re-etiquetar cuando las cosechas maduren.

## 2. Particularidades del ejercicio
- Separadores distintos (`;` y `,`) y `fecha_apertura` en dos formatos (82 filas `DD-MM-YYYY`).
- 49 ingresos corruptos (p. ej. `13.760.566.878.899.200`), recuperados con una regla sin ambigüedad.
- 24 montos solicitados capturados en miles, confirmados contra el saldo del primer corte.
- 122 créditos fantasma (`CAND-PH####`) en cartera sin solicitud → excluidos.
- Escala de mora mixta (buckets hasta 90, días de 91 a 120).
- Pico anómalo de edad = 18 (211 casos vs ≈ 69 en 19–22).
- Ventana de observación desigual (3 a 6 cortes) sin relación con la cosecha.
- Quiebre en datos de pago alternativos desde oct-2024 (utility CSI 0,80) junto con aumento del default.
- El buró se comporta al revés de lo esperado: con registro el default es 15,7 % vs 6,3 %.

## 3. Variables más predictivas y poblaciones
Por IV: recargas mensuales (0,49), pagos alternativos combinados (0,23), registro en buró (0,20)
y % P2P (0,09). Sentido de negocio: en thin-file la capacidad de pago no está documentada; las
recargas son **flujo de caja observado** y la puntualidad en servicios es el **hábito de pago**
más cercano a un historial crediticio. Un % P2P alto sugiere ingresos informales.

**Sí hay dos poblaciones:** con y sin buró tienen perfiles casi idénticos, pero el riesgo con
buró es ≈ 2,5 veces mayor en todos los quintiles (curvas paralelas). Lo interpreto como
**endeudamiento previo**. Por eso usé un modelo único con `has_bureau`; los modelos segmentados
rindieron peor (AUC test 0,74 vs 0,77).

## 4. Tasa de default y desbalance
**10,56 %** (415 de 3.930): 9,5 % en train/test y 13,7 % en OOT. Desbalance moderado (≈ 1:8,5)
que no requiere sobremuestreo (que además distorsionaría la calibración de la PD). Decisiones:
partición estratificada, métricas de ranking (AUC, KS, lift) y calibración en vez de *accuracy*,
y un modelo de pocos parámetros por los ≈ 200 malos de train (LightGBM sobreajustó: 0,87 vs 0,76).

**Con 3 % de default** (≈ 120 malos): ampliaría la ventana o usaría un evento más temprano para
ganar malos, usaría validación cruzada repetida, bins más gruesos y regularización (o pesos por
clase y recalibración), reportaría AUC-PR y lift en el top 5–10 %, y los intervalos de confianza
serían centrales en la decisión.

## 5. KS y AUC en validación

| Muestra | AUC (IC 95 %) | Gini | KS |
|---|---|---|---|
| Test | **0,772** [0,718–0,822] | 0,543 | **0,444** |
| OOT | **0,786** [0,747–0,822] | 0,571 | **0,456** |

Por variable (AUC / KS en test): recargas 0,72 / 0,34; buró 0,62 / 0,24; pagos alternativos
0,58 / 0,13; engagement 0,55 / 0,08; P2P 0,52 / 0,05. El modelo gana ≈ 5 pp de AUC sobre la mejor
variable sola. Brecha train–test de 0,014: el modelo generaliza.

**Rango aceptable:** AUC ≥ 0,70 y KS ≥ 0,30 para un piloto (con el límite inferior del IC por
encima de 0,70); ≥ 0,75 / 0,40 para escalar. En originación con datos alternativos, 0,70–0,80 es
lo usual; por debajo de 0,65 el modelo aporta poco frente a reglas simples y por encima de 0,85
sospecharía *leakage*.

## 6. PSI entre splits
PSI del score: train vs test **0,018** (estable); train vs OOT **0,109** (cambio moderado). La
deriva viene de `alt_pay_avg` (CSI 0,34): el % de clientes en el peor bin de pago pasa de 11 %
a 27 %. Las demás variables son estables (CSI < 0,01).

**Decisión:** aprobar el modelo con condiciones. El ordenamiento se mantiene (AUC OOT 0,79), pero
el nivel estaba subestimado (observado/predicho 1,11). Por eso: (1) reentrenar con el 100 % y
recalibrar el intercepto (−2,15 → −2,05); (2) validar con el proveedor el cambio en el dato de
servicios públicos; (3) alertas de PSI en 0,10 (revisión) y 0,25 (acción).

## 7. Tratamiento del segmento sin buró
El nulo de `bureau_score` es **estructural** (coincide al 100 % con `has_bureau_record = FALSE`),
no un faltante: imputarlo inventaría información. Dentro de quienes lo tienen, el score no
discrimina (AUC 0,51) y su IV aparente venía casi todo del bin "Missing". Por eso lo excluí y usé
el indicador `has_bureau`. Las demás variables son alternativas y están disponibles para todos.
Resultado: AUC en sin buró 0,77 (test) y 0,79 (OOT), mejor que en con buró (0,72–0,73).

## 8. Tres riesgos operativos en producción
1. **Calidad de los datos de entrada** (formatos mixtos, ingresos corruptos, montos en miles,
   cambio en utility). *Mitigación:* validaciones automáticas antes de calificar (tipos, rangos,
   formatos, % de nulos), cuarentena de registros inválidos y binning versionado como artefacto
   para que la transformación sea idéntica en entrenamiento y producción.
2. **Deriva de población y de calibración** (PSI OOT 0,11, default 9,5 % → 13,7 %).
   *Mitigación:* proceso mensual que calcule PSI del score, CSI por variable, mora temprana por
   cosecha y observado/predicho por segmento, con alertas y un procedimiento de recalibración.
3. **Desfase de la etiqueta y sesgo de selección** (el default se confirma 3 meses después y
   sólo se observa a los aprobados). *Mitigación:* etiquetado reproducible con la misma regla,
   indicadores tempranos mientras madura la etiqueta y un grupo de control aprobado bajo el
   corte para *reject inference*.

## 9. Fuera de alcance y qué haría con una semana más
Fuera de alcance: *reject inference* (sólo 70 rechazos sin desempeño), validar la definición
con cosechas de 12 meses, interacciones formales, optimización del corte con rentabilidad,
códigos de razón automáticos y pruebas unitarias.

Con una semana más: validación temporal *rolling* con varias ventanas OOT, análisis de
rentabilidad por corte y segmento, binning óptimo con restricciones y un retador GBM con SHAP,
investigar con el dueño del dato el quiebre de utility, y empaquetar el pipeline con pruebas.

## 10. Uso de IA
Sí, utilicé un asistente de IA (Claude) durante todo el desarrollo de la prueba, como apoyo
y guía paso a paso. Lo usé principalmente para:

- **Rapidez en la generación de código:** el asistente proponía el código de cada sección
  (limpieza de datos, construcción del default, EDA, binning WoE, selección de variables,
  modelamiento y validación), lo que me permitió avanzar más rápido y dedicar más tiempo
  al análisis de los resultados.
- **Estructura del trabajo:** me ayudó a organizar el notebook en secciones y subsecciones
  numeradas, y el repositorio en carpetas (datos, notebook, funciones, modelos y reportes).
- **Redacción de textos:** apoyo en la redacción de las descripciones de cada celda, los
  hallazgos de cada sección, el resumen ejecutivo y estas respuestas.
- **Visualizaciones:** apoyo para construir gráficas más claras, visibles y fáciles de
  interpretar (curvas de cosecha, tasas de default por quintil, estabilidad por bin,
  curvas ROC/KS, calibración y curva estratégica).

Mi trabajo consistió en ejecutar cada celda en mi entorno, revisar e interpretar las
salidas, verificar que los números de los hallazgos coincidieran con lo que obtenía y
corregir errores cuando aparecieron. Las decisiones de modelamiento, como la definición del default, el tratamiento del segmento sin buró, la exclusión del score de buró, el criterio de selección de
variables y la recalibración del modelo, las validé contra la evidencia de mis propios datos antes de adoptarlas.