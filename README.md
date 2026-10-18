# Modelo de PD de originación para población thin-file

Prueba técnica — Científica de Datos, Riesgo Crediticio (Nequi).
Construcción y validación de un modelo de probabilidad de incumplimiento para solicitantes sin
historial en buró, usando datos alternativos (recargas, pagos de servicios públicos y telco,
transacciones y uso de la app).

## Resultados principales

| Indicador | Valor |
|---|---|
| Población de modelamiento | 3.930 créditos desembolsados |
| Definición de default | Mora ≥ 60 días en MOB 1–3 |
| Tasa de default | 10,56 % (train 9,5 % · test 9,5 % · OOT 13,7 %) |
| Modelo | Scorecard: regresión logística sobre WoE, 5 variables |
| AUC / KS test | 0,772 [IC95 0,718–0,822] / 0,444 |
| AUC / KS OOT (oct–dic 2024) | 0,786 [IC95 0,747–0,822] / 0,456 |
| PSI del score train→test / train→OOT | 0,018 (estable) / 0,109 (moderado) |

Variables: `avg_monthly_topups`, `has_bureau`, `alt_pay_avg`, `pct_p2p_transfers`,
`digital_engagement_score`.

## Estructura

    ├── data/
    │   ├── raw/          # archivos originales
    │   └── processed/    # train.csv, test.csv, oot.csv, abt_modelamiento.csv, scores_por_split.csv
    ├── notebooks/
    │   └── 01_modelo_pd_thin_file.ipynb   # notebook principal (ejecutado)
    ├── src/
    │   └── risk_utils.py  # KS, AUC, PSI/CSI, binning WoE, IV, deciles, bootstrap
    ├── models/            # scorecard validado y final (recalibrado)
    ├── reports/           # resumen ejecutivo, respuestas, scorecard, métricas
    └── requirements.txt

## Contenido del notebook
1. Carga · 2. Calidad de datos de solicitudes · 3. Calidad de datos de cartera ·
4. Variable default · 5. EDA · 6. Feature engineering · 7. División y estabilidad ·
8. Binning WoE e IV · 9. Selección de variables · 10. Modelamiento · 11. Validación ·
12. Scorecard y estrategia · 13. Equidad · 14. Modelo final · 15. Conclusiones

## Cómo reproducir

    python -m venv .venv
    .venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    jupyter lab

Abrir `notebooks/01_modelo_pd_thin_file.ipynb` y ejecutar todo. Semilla fija (`SEED = 42`).

## Entregables
- `notebooks/01_modelo_pd_thin_file.ipynb`: análisis, modelo y validación.
- `data/processed/`: división del dataset.
- `reports/resumen_ejecutivo.md`: resumen para stakeholder no técnico.
- `reports/respuestas_preguntas.md`: respuestas a las preguntas de criterio.