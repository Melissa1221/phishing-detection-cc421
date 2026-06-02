# Declaracion de Uso de Inteligencia Artificial (DUIA)

Curso: CC421 Inteligencia Artificial - UNI-FC
Entregable: Avance Preliminar del Proyecto Final (75%)
Proyecto: Deteccion de Phishing y Spear-Phishing en correos

## Herramientas

Usamos asistentes de IA generativa como apoyo durante el desarrollo.

## Alcance del uso

| Actividad | Uso de IA | Validacion humana |
|---|---|---|
| Estructura del repo y scaffolding de modulos | Apoyo | Si |
| Redaccion y documentacion (docstrings, README, informe) | Apoyo | Si |
| Depuracion de errores de entorno y dependencias | Apoyo | Si |
| Diseno experimental (modelos, metricas, particiones) | Equipo | - |
| Ejecucion del pipeline y obtencion de metricas | Equipo | Si |
| Analisis de resultados (deteccion del confound de fuente unica) | Equipo | Si |
| Verificacion de que las metricas reportadas son de la corrida real | Equipo | Si |

## Integridad

- Todas las metricas del informe salen de correr de verdad
  python -m src.run_pipeline sobre el corpus Enron-Spam y son reproducibles.
- El equipo reviso, entiende y se hace responsable del codigo y el texto.
- Las limitaciones (validez externa por el ham de fuente unica) se reportan con
  honestidad y no se esconden detras de las metricas altas.

Integrantes: Estacio Sanchez, Ortega Turpo, Lerzundi Rios, Vega Bendezu, Iman Noriega.
