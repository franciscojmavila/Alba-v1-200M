<img width="100%" alt="Alba Banner GitHub - 14,4K UHD" src="https://github.com/user-attachments/assets/f9bd4d94-46ff-462f-b92e-1046e9546b2b" />

<div align="center">
  
  [![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/franciscojmavila/Alba-v1-200M)
  [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

# Alba v1-200M

Doy por finalizado el modelo Alba v1-200M, un modelo de lenguaje de 208.95M de parámetros construido desde cero. Su arquitectura, tokenizador, pre-entrenamiento y post-entrenamiento fueron desarrollados y entrenados íntegramente sobre una TPU v5e-8 de Google.

Este proyecto nació como un ejercicio personal para entender en la práctica cómo se diseña, escala y distribuye una arquitectura Transformer moderna sin depender de librerías de alto nivel que oculten la complejidad del hardware. Pero también nació del deseo de poder ayudar a mucha gente en el futuro.

## ¿Qué es Alba?

Alba es un proyecto personal y experimental: un modelo de lenguaje bilingüe en español e inglés. Fue construido íntegramente desde cero por mí, un estudiante de ingeniería en computación de la Universidad Nacional de Rafaela (UNRaf), como ejercicio de aprendizaje profundo en arquitecturas Transformer, distribución de entrenamiento en TPU y las técnicas modernas de alineación de modelos (SFT + DPO).

Los datasets de ajuste fino (SFT) y de alineación (DPO) también los creé a mano, supervisando línea por línea. Su personalidad, sus formas de ser y de escribir están basadas en una mujer real. Es por eso que Alba es un modelo más cercano al ser humano, a diferencia de los comerciales que son fríos y corporativos.

## Arquitectura

| Parámetro | Valor |
|---|---|
| Parámetros totales | ~209M (208.95M) |
| Capas | 21 |
| Dimensión oculta | 768 |
| Cabezales de atención | 12 |
| Cabezales clave/valor (GQA) | 4 |
| Dimensión MLP | 2048 |
| Dimensión por cabezal | 64 |
| Ventana de contexto | 1024 tokens |
| Vocabulario | 50.008 tokens (50.000 base + 8 tokens ChatML, con relleno a múltiplo de 8 para el sharding) |

## Tecnologías del pre-entrenamiento

- **JAX + Flax** como stack de entrenamiento, compilado a XLA.
- Arquitectura Transformer decoder-only con **Grouped Query Attention** (GQA, ratio 12:4).
- **Rotary Position Embeddings** (RoPE).
- **SwiGLU** en las capas MLP.
- **RMSNorm** con pre-normalización.
- Vectorización de las 21 capas mediante **`nn.scan`** (compila un solo grafo XLA en
  lugar de 21 grafos independientes).
- **Gradient checkpointing** (rematerialización) para reducir el uso de memoria HBM.
- **Paralelismo de datos totalmente fragmentado (FSDP)** sobre los 8 chips de la TPU.
- Optimizador **AdamW** con warmup lineal y decaimiento coseno.
- Checkpointing asíncrono con **Orbax** (formato Zarr3).
- Tokenizador **BPE a nivel de byte**, bilingüe entre español e inglés, entrenado desde cero,
  con un vocabulario base de 50.000 tokens.
- Streaming de datasets masivos vía Hugging Face Datasets, con muestreo calibrado
  entre fuentes.

### Fuentes de datos del pre-entrenamiento

- Wikipedia (inglés y español).
- Project Gutenberg (literatura clásica en inglés).
- CulturaX (corpus web en español).
- Wikcionario.
- OPUS-100 (corpus paralelo en inglés y español).
- The Stack v2 (código fuente en Python y C).
- CodeSearchNet (Python y Java).
- ArXiv (papers de machine learning).
- Stack Exchange / StackOverflow.
- OpenOrca.

Volumen total: ~14.9 mil millones de tokens procesados a lo largo de 113.500 pasos
de entrenamiento.

## Tecnologías del post-entrenamiento

- Expansión del tokenizador con tokens especiales de formato **ChatML**
  (`<|im_start|>`, `<|im_end|>`, `<|think|>`, `<|/think|>`, `<|answer|>`, `<|/answer|>`).
- Mutación de tensores: expansión de la matriz de embeddings y la cabeza de salida
  con inicialización de ruido gaussiano.
- **Supervised Fine-Tuning (SFT)**:
  - Masked Cross-Entropy Loss. El gradiente solo se propaga sobre los tokens
    generados por el asistente.
  - Sequence Packing. Empaquetado de múltiples conversaciones en una sola
    secuencia de 1024 tokens para maximizar el uso de la TPU.
  - Block-Diagonal Attention Mask. Evita que conversaciones empaquetadas juntas
    se mezclen entre sí.
  - Truncado inteligente que preserva las etiquetas de cierre de turno.
- **Direct Preference Optimization (DPO)**:
  - Función de pérdida de Bradley-Terry.
  - Modelo de referencia congelado (vía `stop_gradient`) para comparar contra el
    modelo activo.
- Exportación a formato **safetensors** (float16).
- Checkpointing y respaldo automático en Hugging Face Hub.

### Datos del post-entrenamiento

- **SFT**: 4.672 conversaciones de entrenamiento + 748 de validación, creadas
  específicamente para este proyecto (personalidad, física, química, código,
  conversación cotidiana bilingüe).
- **DPO**: 5.464 pares de preferencia (chosen/rejected) + 597 de validación.

## Resultados y limitaciones

Las curvas de entrenamiento son sanas. El loss de SFT baja de forma consistente y
el de DPO satura con una precisión superior al 99 % distinguiendo respuestas
preferidas de rechazadas.

Pero hay que ser realista: **Alba tiene una sintaxis, escritura, gramática, manejo de idiomas y
una personalidad bastante fuertes para su tamaño, pero tiene una memoria muy débil y un cerebro muy chico para la gran entropía de datos que recibió.** Con casi 209M de parámetros
expuestos a un corpus de pre-entrenamiento enorme y diverso (código, literatura,
ciencia, filosofía), el modelo no tiene capacidad física para retener hechos
específicos de dominios como física, historia o biología. Sabe estructurar una
oración fluida en español o inglés, mantiene el tono y la personalidad del dataset
de ajuste fino, pero cuando se le pide un dato concreto como, por ejemplo, una fecha, un concepto
técnico, una explicación directa, alucina con total fluidez gramatical.

Esto es un límite físico esperable en modelos de este tamaño, no un error de la arquitectura ni del entrenamiento.
**Alba es un experimento personal y un juguete técnico, no una fuente confiable de información.**

## Gráficos de entrenamiento

**Pre-entrenamiento** (113.500 pasos, loss vs. perplejidad)

<img width="100%" alt="pre-entrenamiento" src="https://github.com/user-attachments/assets/36ba1f27-83d5-4ed2-881d-0a817cd38820" />

**Supervised Fine-Tuning** (80 pasos, Masked CE Loss vs. perplejidad)

<img width="100%" alt="curva_sft" src="https://github.com/user-attachments/assets/d3056c66-851d-43f4-8b20-6c9580579a20" />

**Direct Preference Optimization** (682 pasos, DPO Loss vs. accuracy)

<img width="100%" alt="curva_dpo" src="https://github.com/user-attachments/assets/81ac3308-a610-4078-9729-6ebf0a49a5b2" />

## Requisitos

Para utilizar a Alba no es necesario hardware especializado, ya que es un modelo ligero.

Se necesita como mínimo:
- **Python**: 3.9 o superior.
- **Procesador (CPU)**: 2 núcleos / 2 hilos a 2.0GHz o más (cualquier procesador x86_64 o ARM64 moderno de gama de entrada).
- **Memoria RAM**: 2 GB libres (los pesos ocupan ~400 MB en disco y ~800 MB en RAM al cargarse en float32, más lo que consume JAX para la compilación JIT).
- **Espacio en disco**: ~1GB.
- **Tarjeta gráfica**: No requerida. No hace falta GPU dedicada ni gráfica integrada (iGPU). La inferencia se ejecuta íntegramente sobre la CPU.


**Por el momento no es posible ejecutarla en celulares.**

## Cómo descargar y probar el modelo

### 1. Descargar los archivos

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path "$HOME\Downloads\Alba-v1-200M" | Out-Null
cd "$HOME\Downloads\Alba-v1-200M"
Invoke-WebRequest "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/alba-v1.py" -OutFile "alba-v1.py"
Invoke-WebRequest "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/Alba-v1-200M_config.json" -OutFile "Alba-v1-200M_config.json"
Invoke-WebRequest "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/tokenizador_bilingue-vocab.json" -OutFile "tokenizador_bilingue-vocab.json"
Invoke-WebRequest "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/tokenizador_bilingue-merges.txt" -OutFile "tokenizador_bilingue-merges.txt"
Invoke-WebRequest "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/Alba-v1-200M.safetensors" -OutFile "Alba-v1-200M.safetensors"
```

**Linux / macOS (Terminal):**
```bash
cd "$(xdg-user-dir DOWNLOAD 2>/dev/null || echo "$HOME/Downloads")"
mkdir -p Alba-v1-200M
cd Alba-v1-200M
curl -LO "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/alba-v1.py"
curl -LO "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/Alba-v1-200M_config.json"
curl -LO "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/tokenizador_bilingue-vocab.json"
curl -LO "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/tokenizador_bilingue-merges.txt"
curl -LO "https://huggingface.co/franciscojmavila/Alba-v1-200M/resolve/main/Alba-v1-200M.safetensors"
```

### 2. Crear entorno virtual e instalar dependencias

**Windows (PowerShell):**
```powershell
python -m venv Alba
.\Alba\Scripts\Activate.ps1
pip install jax jaxlib flax safetensors tokenizers numpy
```

**Linux / macOS (Terminal):**
```bash
python3 -m venv Alba
source Alba/bin/activate
pip install jax jaxlib flax safetensors tokenizers numpy
```

### 3. Ejecutar

```bash
python alba-v1.py
```

El archivo `alba-v1.py` ya contiene toda la arquitectura y la lógica de inferencia. No se recomienda alterar los hiperparámetros de generación, ya que fueron calibrados específicamente para esta versión del modelo. Modificarlos puede producir más alucinaciones o bucles de repetición.

> **Nota:** Si movés o renombrás la carpeta `Alba-v1-200M` después de haber creado el entorno virtual, vas a tener que recrearlo (borrar la carpeta `Alba` y repetir el paso 2). Esto es una limitación de Python, no de Alba.

# Agradecimiento

Estoy muy feliz si descargaste este modelo y al menos lo probaste una sola vez.

El origen de este proyecto fue en noviembre de 2025, y se extendió hasta finales de julio de 2026, luego de 5 arduos años de estudio en modelos de inteligencia artificial. Durante estos nueve meses la meta fija fue crear una herramienta de accesibilidad real, gratuita y privada para personas con problemas motrices como, por ejemplo, personas que no pueden usar sus manos y están limitadas a no poder usar una computadora sin asistencia humana en esta era moderna. La visión para las siguientes versiones de Alba es convertirla en un agente capaz de manipular el sistema operativo, interpretar el entorno y ejecutar acciones mediante comandos de voz.

Para construir un agente que controle una computadora con exactitud, primero necesitaba dominar y comprender el cerebro del modelo. Esta versión de 200M es ese paso técnico fundamental. Nació de la necesidad de entender en la práctica cómo se diseña, escala y distribuye una arquitectura Transformer moderna sin usar librerías de alto nivel que oculten la complejidad real del hardware.

Lamento si Alba ahora mismo no es lo que esperabas. Prometo mejorarla en las próximas versiones, es un hecho.

De verdad, muchas gracias.

## Licencia

Este proyecto se distribuye bajo licencia **CC BY-NC 4.0** (Creative Commons
Attribution-NonCommercial 4.0 International).

Cualquier persona o empresa puede usar Alba libremente. No
permito que se venda ni se use con fines de lucro, ni que se use sin dejar
constancia de quién lo creó. Pero si lo usás para ayudar a alguien, con fines
benéficos o educativos sin ánimo de lucro, tenés permiso total y garantizado.

Los datos de pre-entrenamiento provienen de fuentes con licencias públicas
(Wikipedia bajo CC-BY-SA, Project Gutenberg de dominio público, The Stack v2 bajo
las licencias originales de cada repositorio). Los pesos del modelo se publican
bajo la licencia elegida para este repositorio.

---

<div align="right">
  
  *Alba, versión 1. 200M | nov.2025–jul.2026*

  *Contacto: francisco.avila.dev@gmail.com*
  
  *Francisco Ávila | Ingeniería en Computación*
  
</div>
