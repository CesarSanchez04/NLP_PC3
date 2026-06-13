# Práctica Calificada 3 - CC0C2 Procesamiento del Lenguaje Natural
## Proyecto 5: Continual Pre-Training, Replay y Olvido Catastrófico

Este repositorio contiene la solución completa y reproducible para el **Proyecto 5** de la Práctica Calificada 3 de la materia de NLP. Nos enfocamos en estudiar de forma práctica el fenómeno de olvido catastrófico (*catastrophic forgetting*) al hacer preentrenamiento continuo (*continual pre-training*) de un modelo de lenguaje y cómo podemos usar la repetición de memoria (*memory replay*) para mitigar este efecto.

---

## 1. Objetivo del Proyecto

El objetivo principal es demostrar con datos y código cómo adaptar un modelo causal autoregresivo (`distilgpt2`) a un nuevo dominio técnico especializado (alineamiento de modelos) degrada su rendimiento y conocimiento en su dominio base original. Evaluaremos cómo aplicar la técnica de *Memory Replay* ayuda a que el modelo retenga su conocimiento previo.

---

## 2. Cuaderno Base y Estructura

- **Cuaderno Base:** [Cuaderno16-CC0C2.ipynb](file:///Users/cesar/Desktop/CC-0C2/Semana9/Cuaderno16-CC0C2.ipynb)
- **Notebook Solución:** [Proyecto5_Continual_Learning.ipynb](file:///Users/cesar/Desktop/CC-0C2/Semana9/Proyecto5_Continual_Learning.ipynb)

Estructura del notebook:
1. **Identificación y Celda de Verificación Personal:** Información de control del estudiante y parámetros de ejecución.
2. **Marco Teórico:** Explicación simplificada de continual learning, olvido catastrófico y replay.
3. **Carga y Configuración del Modelo:** Configuración de `distilgpt2` y tokenizador (padding e ignore_index en -100).
4. **Construcción del Corpus:**
   - *Dominio Base (Viejo):* 10 ejemplos de cultura general, geografía y programación clásica en Python.
   - *Dominio Nuevo (Específico):* 10 ejemplos sobre teoría moderna de alineamiento y PEFT (DPO, ORPO, LoRA, PEFT).
5. **Implementación de Memory Replay:** Función `mix_replay` con proporción ajustable.
6. **Métricas de Evaluación Interna:** Cálculo de pérdida causal y decodificador para generación cualitativa.
7. **Tabla de Dimensiones y Evidencia de Cálculo Interno:** Detalle de las dimensiones de tensores y código en PyTorch para calcular manualmente la entropía cruzada con desplazamiento causal.
8. **Configuración Experimental (Línea Base vs Variante):**
   - *Línea Base:* Entrenamiento con 100% de datos del dominio nuevo.
   - *Variante:* Entrenamiento con datos mezclados con replay ratio de 0.5.
9. **Comparación Cuantitativa y Cualitativa:** Tabla comparativa de pérdidas e inferencia de texto con prompts específicos antes y después.
10. **Preguntas Avanzadas Obligatorias.**
11. **Preguntas Transversales Seleccionadas.**
12. **Conclusiones y Puente al Curso.**

---

## 3. Línea Base y Modificación Realizada

- **Línea Base:** Se entrena el modelo base `distilgpt2` por 20 épocas utilizando únicamente los textos del **Dominio Nuevo** (conceptos técnicos). Al no recibir datos del dominio original, el optimizador ajusta los pesos de forma libre para la nueva tarea, lo que causa olvido catastrófico.
- **Variante (Modificación Realizada):** Se realiza el mismo entrenamiento con idéntica semilla e hiperparámetros, pero usando un lote mixto generado con la función `mix_replay(texts_new, texts_base, replay_ratio=0.5)`. Esto añade ejemplos del dominio viejo (50% de la cantidad de ejemplos nuevos) para balancear la actualización de los gradientes.

---

## 4. Cómo Ejecutar el Notebook

1. **Entorno Virtual:**
   - Para que el código corra sin errores, asegúrate de activar el entorno virtual local de tu workspace:
     ```bash
     source ../.venv/bin/activate
     ```
2. **Abrir y Ejecutar:**
   - Abre el notebook en Jupyter:
     ```bash
     jupyter notebook
     ```
   - Abre [Proyecto5_Continual_Learning.ipynb](file:///Users/cesar/Desktop/CC-0C2/Semana9/Proyecto5_Continual_Learning.ipynb) y corre las celdas de forma secuencial.

---

## 5. Resumen de Resultados Obtenidos

| Modelo | Pérdida en Dominio Viejo (Base) | Pérdida en Dominio Nuevo |
|---|---|---|
| **Modelo Base Inicial** | **~5.12** | **~5.31** |
| **Sin Replay (Línea Base)** | **~7.88** *(Se eleva: Olvido)* | **~1.12** *(Adaptado)* |
| **Con Replay (Variante)** | **~5.19** *(Estable: Protegido)* | **~1.24** *(Adaptado)* |

*Análisis:* El modelo afinado sin regularización degrada fuertemente su conocimiento previo. Al activar un 50% de replay, la pérdida en el dominio base se mantiene estable (~5.19 frente a ~5.12 inicial) y el modelo logra aprender de forma efectiva la nueva jerga técnica.

---

## 6. Limitaciones Identificadas

1. **Capacidad del Modelo:** Al usar `distilgpt2` (82M parámetros), la capacidad de retención y representación semántica en simultáneo es pequeña comparado con LLMs modernos.
2. **Tamaño del Corpus:** Usamos un dataset sintético reducido (10 ejemplos por dominio). Un entrenamiento de producción real requeriría miles de ejemplos representativos muestreados en un pipeline continuo.
3. **Selección Estática:** El replay se hace a partir de una lista de ejemplos estática en lugar de un buffer dinámico basado en reservorios con similitud semántica.

---

## 7. Qué mostrar en el Video de Defensa (Mínimo de 12 minutos)

Sigue la estructura definida en el archivo [speech_video.md](file:///Users/cesar/Desktop/CC-0C2/Semana9/speech_video.md):
1. **Presentación (1-2 min):** Muestra tu rostro y detalla los objetivos del proyecto.
2. **Marco Teórico (2 min):** Explica qué es el olvido catastrófico y cómo opera el memory replay.
3. **Codificación en Vivo A (2-3 min):** 
   - Cambia la `temperature` (ej. de `0.8` a `0.1`) en la celda de generación cualitativa.
   - Explica el impacto matemático que esperas (menor entropía, salida más determinista), ejecuta en vivo y comenta el resultado del texto obtenido.
4. **Codificación en Vivo B (3-4 min):**
   - Cambia el `replay_ratio` (ej. de `0.5` a `0.1`) en la variante con replay.
   - Explica que con menor tasa el olvido será mayor (elevará la pérdida en el dominio viejo), corre el entrenamiento rápido en vivo y muestra cómo se afecta la pérdida en la tabla final.
5. **Cálculo Interno e Inferencia (2 min):** Explica la forma de los tensores de logits `[Batch, Seq, Vocab]` y el desplazamiento a la izquierda de las etiquetas para calcular la pérdida causal.
6. **Preguntas Técnicas Avanzadas y Transversales (3 min).**

---

## 8. Qué hice, por qué lo hice y qué significan mis resultados

- **Qué hice:** Desarrollé un marco experimental comparativo en PyTorch para medir y mitigar el olvido catastrófico en `distilgpt2` adaptando su corpus de lenguaje al dominio técnico de alineamiento de LLMs, con y sin buffers de replay.
- **Por qué lo hice:** Para aislar empíricamente a nivel de gradientes por qué una red neuronal tiende a sobreescribir su memoria y comprobar de qué manera inyectar datos previos regulariza el aprendizaje.
- **Qué significan mis resultados:** Validamos que la mezcla balanceada de datos antiguos (replay ratio de 0.5) protege casi al 100% el conocimiento lingüístico previo del modelo sin afectar de manera drástica su capacidad de asimilar conceptos de dominios altamente especializados.

---

## 9. Puente al Curso

- **PEFT (LoRA / QLoRA):** Vimos que afinar todos los pesos causa olvido catastrófico global. LoRA es una alternativa estructural en la que se congela $W_0$ y solo se entrenan adaptadores de bajo rango. Al no modificar los pesos originales, el conocimiento base queda protegido sin requerir buffers de replay.
- **Alineamiento (DPO):** Al entrenar DPO con preferencias de usuario, se añade un término de penalización KL con respecto a un modelo de referencia. Esto actúa como un replay virtual suavizado, evitando que la política de alineamiento decolore o pierda las capacidades del modelo base genérico.

---

## 10. Declaración de Autoría y Uso de IA

```text
Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta práctica. 
Si utilicé herramientas de IA, las usé como apoyo para redacción, depuración o consulta, pero la implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mía.
```
*Uso de IA:* Se usó apoyo de IA únicamente para estructurar y dar formato a este archivo README técnico y para optimizar el script generador de celdas de Jupyter.
