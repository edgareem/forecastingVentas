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

`inferencia_df`:
['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria', 'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta', 'ingresos', 'anio', 'trimestre', 'mes', 'nombre_mes', 'dia', 'dia_anio', 'semana_anio', 'dia_semana_num', 'dia_semana', 'es_fin_de_semana', 'es_inicio_mes', 'es_fin_mes', 'es_inicio_trimestre', 'es_fin_trimestre', 'es_inicio_anio', 'es_fin_anio', 'es_primera_quincena', 'es_ultima_quincena', 'es_payday_inicio_mes', 'es_payday_fin_mes', 'es_festivo', 'nombre_festivo', 'es_vispera_festivo', 'es_post_festivo', 'es_black_friday', 'es_cyber_monday', 'dias_hasta_fin_mes', 'semana_mes', 'temporada_rebajas_invierno', 'temporada_rebajas_verano', 'campana_navidad', 'campana_vuelta_al_cole', 'es_puente', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'media_movil_7_dias', 'descuento_porcentaje', 'precio_competencia', 'ratio_precio', 'nombre_h_Adidas Own The Run Jacket', 'nombre_h_Adidas Ultraboost 23', 'nombre_h_Asics Gel Nimbus 25', 'nombre_h_Bowflex SelectTech 552', 'nombre_h_Columbia Silver Ridge', 'nombre_h_Decathlon Bandas Elásticas Set', 'nombre_h_Domyos BM900', 'nombre_h_Domyos Kit Mancuernas 20kg', 'nombre_h_Gaiam Premium Yoga Block', 'nombre_h_Liforme Yoga Pad', 'nombre_h_Lotuscrafts Yoga Bolster', 'nombre_h_Manduka PRO Yoga Mat', 'nombre_h_Merrell Moab 2 GTX', 'nombre_h_New Balance Fresh Foam X 1080v12', 'nombre_h_Nike Air Zoom Pegasus 40', 'nombre_h_Nike Dri-FIT Miler', 'nombre_h_Puma Velocity Nitro 2', 'nombre_h_Quechua MH500', 'nombre_h_Reebok Floatride Energy 5', 'nombre_h_Reebok Professional Deck', 'nombre_h_Salomon Speedcross 5 GTX', 'nombre_h_Sveltus Kettlebell 12kg', 'nombre_h_The North Face Borealis', 'nombre_h_Trek Marlin 7', 'categoria_h_Fitness', 'categoria_h_Outdoor', 'categoria_h_Running', 'categoria_h_Wellness', 'subcategoria_h_Banco Gimnasio', 'subcategoria_h_Bandas Elásticas', 'subcategoria_h_Bicicleta Montaña', 'subcategoria_h_Bloque Yoga', 'subcategoria_h_Cojín Yoga', 'subcategoria_h_Esterilla Fitness', 'subcategoria_h_Esterilla Yoga', 'subcategoria_h_Mancuernas Ajustables', 'subcategoria_h_Mochila Trekking', 'subcategoria_h_Pesa Rusa', 'subcategoria_h_Pesas Casa', 'subcategoria_h_Rodillera Yoga', 'subcategoria_h_Ropa Montaña', 'subcategoria_h_Ropa Running', 'subcategoria_h_Zapatillas Running', 'subcategoria_h_Zapatillas Trail']

Columnas one hot finales creadas a partir de variables categóricas:
- `nombre_h_...`
- `categoria_h_...`
- `subcategoria_h_...`

