# Forecasting Ventas

Proyecto base de data science en Python para experimentacion, entrenamiento de modelos y despliegue de una app sencilla.

## Estructura

```text
forecastingVentas/
|-- app/                # Aplicacion o dashboard
|-- data/
|   |-- external/       # Datos externos de terceros
|   |-- interim/        # Datos temporales transformados
|   |-- processed/      # Datos listos para modelado
|   `-- raw/            # Datos originales
|-- docs/               # Documentacion del proyecto
|-- models/             # Modelos entrenados y artefactos
|-- notebooks/          # Notebooks de analisis
|-- src/
|   `-- forecasting_ventas/
|       |-- data/       # Carga e ingestion
|       |-- features/   # Ingenieria de variables
|       |-- models/     # Entrenamiento e inferencia
|       `-- utils/      # Utilidades compartidas
|-- tests/              # Tests automatizados
|-- requirements.txt
`-- README.md
```

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Flujo sugerido

1. Guardar los datos originales en `data/raw/`.
2. Explorar y validar en `notebooks/`.
3. Mover la logica reutilizable a `src/forecasting_ventas/`.
4. Guardar datasets preparados en `data/processed/`.
5. Entrenar y guardar modelos en `models/`.
6. Cubrir funciones clave con tests en `tests/`.

## Siguientes pasos

- Crear el primer notebook de exploracion.
- Definir la variable objetivo y el horizonte de prediccion.
- Implementar un pipeline base de entrenamiento.
