<!--
Nota para el curso:
Este proyecto usa AGENTS.md como archivo principal de instrucciones para Codex.
Este archivo se conserva como parte de la estructura base, pero las instrucciones
operativas para Codex deben mantenerse en AGENTS.md.
-->

* empieza siempre tu respuesta con el emoji 🤖-
* responde siempre en español
* recuerda que las variables del dataframe `df` que debes usar en el código deben salir de este estado actual del proyecto
* no uses en tu código ninguna otra variable que no esté en la lista siguiente, salvo que la definas tú mismo en el código que generes
* no uses ninguna librería que no sean estas: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, streamlit, holidays

`df`:
['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria', 'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta', 'ingresos', 'anio', 'trimestre', 'mes', 'nombre_mes', 'dia', 'dia_anio', 'semana_anio', 'dia_semana_num', 'dia_semana', 'es_fin_de_semana', 'es_inicio_mes', 'es_fin_mes', 'es_inicio_trimestre', 'es_fin_trimestre', 'es_inicio_anio', 'es_fin_anio', 'es_primera_quincena', 'es_ultima_quincena', 'es_payday_inicio_mes', 'es_payday_fin_mes', 'es_festivo', 'nombre_festivo', 'es_vispera_festivo', 'es_post_festivo', 'es_black_friday', 'es_cyber_monday', 'dias_hasta_fin_mes', 'semana_mes', 'temporada_rebajas_invierno', 'temporada_rebajas_verano', 'campana_navidad', 'campana_vuelta_al_cole', 'es_puente', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'media_movil_7_dias', 'descuento_porcentaje', 'precio_competencia', 'ratio_precio']

Columnas one hot finales creadas a partir de variables categóricas:
- `nombre_h_...`
- `categoria_h_...`
- `subcategoria_h_...`

