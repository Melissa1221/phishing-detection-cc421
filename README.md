# Deteccion de Phishing y Spear-Phishing en correos

Curso CC421 Inteligencia Artificial - Ciencia de la Computacion, Facultad de Ciencias, UNI.
Entregable: Avance Preliminar del Proyecto Final (75%).

Clasificador binario supervisado que separa correos legitimos de maliciosos
(phishing/spam). Usamos representacion TF-IDF y comparamos 4 clasificadores
clasicos: Naive Bayes, Regresion Logistica, Random Forest y SVM lineal. Para
elegir el modelo priorizamos el recall sobre la clase phishing, porque dejar
pasar un ataque (falso negativo) sale mas caro que una falsa alarma.

## Integrantes

| Integrante | Rol |
|---|---|
| Estacio Sanchez, Jose Rodolfo | Modelado y algoritmia (train.py) |
| Ortega Turpo, Junior | Datos y preprocesamiento (data_loader.py, preprocess.py, features.py) |
| Lerzundi Rios, Juan de Dios Fernando | Analisis exploratorio (eda.py) |
| Vega Bendezu, Ale Bruno | Evaluacion y metricas (evaluate.py) |
| Iman Noriega, Melissa | NLP, documentacion e integracion (predict.py, run_pipeline.py) |

## Estructura

```
phishing-detection/
├── README.md
├── requirements.txt
├── DUIA.md                     declaracion de uso de IA (la pide el curso)
├── src/
│   ├── config.py               rutas, semilla, hiperparametros
│   ├── data_loader.py          descarga y consolida el corpus Enron-Spam
│   ├── preprocess.py           limpieza de HTML, tokens especiales, stopwords
│   ├── features.py             TF-IDF + particion 70/15/15
│   ├── train.py                los 4 clasificadores como pipelines
│   ├── evaluate.py             metricas, matriz de confusion, ROC/PR
│   ├── eda.py                  analisis exploratorio (figuras + stats)
│   ├── predict.py              inferencia sobre un correo en bruto
│   └── run_pipeline.py         corre todo el flujo de punta a punta
├── notebooks/
│   └── phishing_detection_demo.ipynb
├── data/
│   ├── raw/                    corpus crudo (se descarga solo)
│   └── processed/              corpus limpio
├── models/                     modelos serializados (best_model.joblib)
└── reports/
    ├── figures/
    └── metrics/
```

## Como correrlo

Probado con Python 3.12 (en Colab funciona con 3.10/3.11).

```bash
git clone <URL-del-repo> && cd phishing-detection

python -m venv .venv
source .venv/bin/activate          # windows: .venv\Scripts\activate
pip install -r requirements.txt

# pipeline completo (la primera vez descarga el dataset)
python -m src.run_pipeline

# para probar rapido con un subconjunto:
python -m src.run_pipeline --sample 5000
```

run_pipeline.py hace en orden: carga del corpus, limpieza, EDA, particion
70/15/15, entrena los 4 modelos, validacion cruzada 5-fold, evalua en
validacion, elige el mejor por F1, evalua una sola vez en test, y guarda
figuras y el mejor modelo.

### Clasificar un correo suelto

```bash
python -m src.predict --text "URGENT: verify your account at http://phish.example.com or it will be suspended"
# Prediccion: PHISHING (prob. phishing = 0.94)
```

## Datos

Para este avance usamos el corpus publico Enron-Spam (Metsis et al., 2006),
unos 33 000 correos etiquetados como spam/ham. Despues de juntar, quitar
correos vacios y deduplicar quedan 30 461 correos (aprox 48% phishing/spam,
52% legitimos). El data_loader esta hecho para meter SpamAssassin y Nazario en
la fase final sin tocar el resto del pipeline.

El corpus se descarga solo la primera vez desde un mirror publico en GitHub. No
se versiona en el repo (ver .gitignore).

## Resultados preliminares

Las metricas de la ultima corrida quedan en reports/metrics/ (val_metrics.json,
test_metrics.json) y las figuras en reports/figures/. Las tres metas de la
propuesta (recall >= 0.90, F1 >= 0.88, PR-AUC >= 0.92 sobre la clase phishing)
se cumplen en el test reservado.

## Reproducibilidad

- Semilla global en config.RANDOM_STATE = 42 (numpy, random y el random_state de
  todos los estimadores y particiones).
- Versiones fijadas en requirements.txt.
- El test se separa al inicio y se evalua una sola vez.

## Lo que falta para el 25% restante

Integrar SpamAssassin y Nazario, comparar TF-IDF contra embeddings
(Word2Vec/GloVe), probar SMOTE frente al re-pesado actual, buscar
hiperparametros con GridSearchCV, y ver interpretabilidad (terminos de mayor
peso por clase).
