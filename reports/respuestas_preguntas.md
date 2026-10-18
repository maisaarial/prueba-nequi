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
buró es ≈ 2,5 veces mayor en promedio (15,7 % vs 6,3 %) y es mayor en todos los quintiles de
las variables principales (curvas paralelas). Lo interpreto como **endeudamiento previo**.

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
variable sola. Brecha train–test de 0,014: el modelo generaliza. En la práctica, el 30 % de solicitudes más riesgosas concentra ≈ 2/3 de los malos (67,9 % en test, 65,9 % en OOT) y el primer decil tiene una tasa de default 3 veces la promedio.

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
servicios públicos; la subestimación se concentra en clientes con buró (observado 20,9 % vs PD 16,9 % en OOT), por lo que la recalibración debe monitorearse por segmento. (3) alertas de PSI en 0,10 (revisión) y 0,25 (acción).

## 7. Tratamiento del segmento sin buró
El nulo de `bureau_score` es **estructural** (coincide al 100 % con `has_bureau_record = FALSE`),
no un faltante: imputarlo inventaría información. Dentro de quienes lo tienen, el score no
discrimina (AUC 0,51) y su IV aparente venía casi todo del bin "Missing". Por eso lo excluí y usé
el indicador `has_bureau`. Las demás variables son alternativas y están disponibles para todos.
Resultado: AUC en sin buró 0,77 (test) y 0,79 (OOT), mejor que en con buró (0,72–0,73).

## 8. Tres riesgos operativos en producción

1. **Calidad y consistencia de los datos de entrada.** Es lo que más falló en este ejercicio:
   fechas en dos formatos, 49 ingresos con formato corrupto, 24 montos en miles y el cambio
   en el dato de servicios públicos. *Mitigación:* validaciones automáticas antes de calificar
   (tipos, rangos, formatos y % de nulos), por ejemplo con *expectations* en Databricks
   (Delta Live Tables) o un paso de validación en el pipeline de SageMaker, con cuarentena
   de registros inválidos. El binning WoE se versiona como artefacto en MLflow para que la
   transformación sea idéntica en entrenamiento y producción.
2. **Deriva de población y de calibración.** En OOT el PSI del score fue 0,11, el CSI de
   `alt_pay_avg` 0,34 y el default subió de 9,5 % a 13,7 %. *Mitigación:* un job mensual de
   monitoreo (SageMaker Model Monitor o un job programado en Databricks) que calcule el PSI
   del score, el CSI por variable, la mora temprana por cosecha y la relación
   observado/predicho por segmento de buró. Alertas en PSI 0,10 (revisión) y 0,25 (acción),
   con un procedimiento definido de recalibración del intercepto.
3. **Desfase de la etiqueta y sesgo de selección.** El default se confirma 3 meses después del
   desembolso y sólo se observa a los aprobados. *Mitigación:* un pipeline de etiquetado
   reproducible (la misma regla de 60+ en MOB 1–3 en SQL/Spark, con control de ventana
   completa), indicadores tempranos mientras madura la etiqueta (mora al MOB 1 y 2) y un
   pequeño grupo de control aprobado bajo el corte para hacer *reject inference* y no
   degradar el modelo en cada reentrenamiento.

## 9. Fuera de alcance y qué haría con una semana más
Fuera de alcance: *reject inference* (sólo 70 rechazos sin desempeño), validar la definición
con cosechas de 12 meses, interacciones formales, optimización del corte con rentabilidad,
códigos de razón automáticos y pruebas unitarias.

Con una semana más: validación temporal *rolling* con varias ventanas OOT, análisis de
rentabilidad por corte y segmento, binning óptimo con restricciones y un retador GBM con SHAP,
investigar con el dueño del dato el quiebre de utility, y empaquetar el pipeline con pruebas.

## 10. Uso de IA
Sí. Utilicé un asistente de IA (Claude) durante todo el desarrollo como apoyo y guía paso a
paso: para agilizar la generación del código de cada sección (limpieza de datos, construcción
del default, EDA, binning WoE, selección de variables, modelamiento y validación), para
organizar el notebook y el repositorio, para apoyar la redacción de las descripciones,
hallazgos, resumen ejecutivo y estas respuestas, y para construir gráficas más claras y
fáciles de interpretar. Mi trabajo consistió en ejecutar cada celda en mi entorno, revisar e
interpretar las salidas, verificar que los números de los hallazgos coincidieran con lo que
obtenía y corregir los errores que aparecieron. Las decisiones de modelamiento (definición del
default, tratamiento del segmento sin buró, exclusión del score de buró, criterio de selección
de variables y recalibración del modelo) las validé contra la evidencia de mis propios datos
antes de adoptarlas.