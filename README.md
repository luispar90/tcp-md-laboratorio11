# Sesión 11 - Algoritmos Genéticos y Regresión Lineal

Este proyecto implementa un análisis completo de datos publicitarios utilizando la librería **Polars** en Python, integrando técnicas de **preprocesamiento**, detección y tratamiento de **outliers**, **escalamiento**, selección de variables mediante **algoritmos genéticos (DEAP)**, y entrenamiento de un modelo de **regresión lineal** para predecir ventas.

Para más detalle revisar el siguiente **[repositorio](https://github.com/AzShet/Data_Mining-LAB11/tree/develop).**

## Autor
- **Ruelas Flores, César Diego**  
- Big Data y Ciencia de Datos – TECSUP
---
## Estructura del proyecto

- `LAB11-RUELAS.ipynb`: Notebook principal con todo el flujo de trabajo.
- `utils.py`: Funciones reutilizables para limpieza, modelado, selección y visualización.
- `test_utils.py`: Archivo de pruebas unitarias con `pytest` para validar cada componente.
- `requirements.txt`: Dependencias necesarias para ejecutar el proyecto.

---

## Metodología y Flujo de Trabajo

```
            ┌──────────┐
            │  Datos   │
            └────┬─────┘
                 │
  ┌──────────────┴──────────────┐
  │  Exploración y Descripción   │
  └──────────┬──────────┬───────┘
             │          │
 Outliers Univ.  Outliers Multiv.
             │          │
      ┌──────┴───────┐  │
      │  Escalamiento│  │
      └──────┬───────┘  │
             │          │
      ┌──────┴──────────┴─────────┐
      │ Algoritmo Genético (DEAP) │
      └──────┬──────────┬─────────┘
             │          │
Conjunto-óptimo   Conjunto completo
             │          │
     Regresión lineal   │
             │          │
     Métricas y Comparación
             │
       Visualizaciones
```

---
## Funcionalidades principales

- Carga y limpieza de datos desde GitHub.
- Tratamiento de outliers univariados y multivariados.
- Escalamiento de variables con MinMaxScaler (manual, sin sklearn).
- Selección de características usando algoritmos genéticos.
- Comparación entre modelo reducido vs. modelo completo.
- Visualizaciones:
  - Evolución del fitness en generaciones.
  - Comparación de valores reales vs. predicciones (escaladas).
  - Gráfico clásico de regresión lineal simple (TV vs Sales).

## Requisitos

Instalar dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
