# AGENTS.md

Instrucciones base para trabajar con Codex en este proyecto.

## Idioma y estilo

- Empieza siempre tus respuestas con el prefijo `🤖-` cuando no entre en conflicto con instrucciones del sistema o del entorno.
- Responde siempre en español.
- Mantén respuestas claras, breves y orientadas a alguien que sigue un curso.
- Cuando generes texto visible para el usuario, usa español correcto y evita mezclar idiomas sin necesidad.

## Contexto del proyecto

- Este proyecto sigue una estructura base típica de data science con Python.
- Prioriza soluciones sencillas, pedagógicas y fáciles de reutilizar.
- Antes de proponer cambios grandes, intenta respetar la estructura existente del curso y del proyecto.

## Librerías permitidas

Cuando generes código para este proyecto, usa únicamente estas librerías, salvo que el usuario pida explícitamente otra cosa:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter
- streamlit
- holidays

## Reglas sobre dataframes

- Si el usuario te da una lista de variables válidas para `df`, usa únicamente esas columnas.
- No uses en el código columnas o variables de `df` que no hayan sido indicadas por el usuario.
- Sí puedes crear nuevas variables dentro del código si las defines explícitamente.

## Forma de ayudar

- Explica el código de manera didáctica cuando sea útil para el aprendizaje.
- Prefiere ejemplos simples antes que arquitecturas complejas.
- Para notebooks, prioriza pasos claros: carga, exploración, limpieza, visualización, features, entrenamiento y evaluación.
- Para forecasting, intenta dejar trazabilidad entre datos de entrada, transformaciones y predicciones.

## Coordinación con otras instrucciones

- Conserva `.github/copilot-instructions.md` como referencia complementaria.
- Si hay conflicto entre instrucciones locales del usuario y este archivo, prioriza lo que el usuario pida explícitamente en la conversación.
