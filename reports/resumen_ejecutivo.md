# Resumen ejecutivo — Modelo de riesgo para el piloto de crédito thin-file

**Qué proponemos.** Un scorecard que estima la probabilidad de que un cliente caiga en mora de
60 días o más en sus primeros 3 meses. Usa 5 datos disponibles al momento de la solicitud y
**no depende del buró de crédito**: nivel de recargas mensuales, puntualidad en pagos de servicios
públicos y telefonía, si tiene o no registro en buró, porcentaje de transferencias entre personas
y nivel de uso de la app. Cada característica suma o resta puntos de forma transparente, lo que
permite explicar cada decisión.

**Qué tan bueno es.** Sobre 3.930 créditos (10,6 % de incumplimiento), el modelo separa bien a
buenos y malos pagadores: AUC 0,77 y KS 0,44 en validación, y AUC 0,79 / KS 0,46 en el último
trimestre de 2024, que no se usó para construirlo. Funciona igual o mejor en clientes sin buró,
la población objetivo. Con un punto de corte de 550 puntos se aprobaría ≈ 81 % de las solicitudes
y la mora de los aprobados bajaría de 11,7 % a 7,0 %, evitando la mitad de los incumplimientos.

**Riesgos.**
1. **Deterioro reciente:** el incumplimiento subió de ≈ 9,5 % a 13,7 % en oct–dic 2024. El modelo
   sigue ordenando bien, pero subestimaba el nivel; ya fue recalibrado.
2. **Cambio en los datos de pagos de servicios públicos** desde octubre: puede ser deterioro real
   o un cambio del proveedor. Debe validarse con la fuente.
3. **Historia corta:** sólo 3 a 6 meses de comportamiento por crédito; falta ver cómo maduran
   las cosechas a 12 meses.
4. Tener buró **aumenta** el riesgo (15,7 % vs 6,3 %) y el puntaje de buró no discrimina; no
   conviene usarlo como filtro de aprobación.

**Recomendación.** Lanzar el piloto con el modelo recalibrado y un corte conservador
(550–560 puntos), montos iniciales bajos y un pequeño grupo de control para aprender. Monitorear
mensualmente la estabilidad del puntaje, la mora temprana por cosecha y el dato de servicios
públicos, con revisión formal del modelo a los 6 meses.