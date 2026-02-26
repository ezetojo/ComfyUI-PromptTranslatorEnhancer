# ComfyUI-PromptTranslator-Enhancer
**Autor**: EzeTojo | **Versión**: 0.0.1 (Nightly)

Custom node para ComfyUI que traduce y mejora prompts desde distintos idiomas al inglés usando un modelo de lenguaje local (GGUF). Los idiomas soportados dependen exclusivamente del modelo que elijas usar.

## Características

- 🌐 **Multi**: Traduce automáticamente desde los idiomas soportados por el modelo al inglés
- ✨ **Mejora de prompts** con 3 niveles: basic, detailed, creative
- 🖥️ **100% local** — usa modelos GGUF vía `llama-cpp-python`
- 🔄 **Modelo reutilizable** — carga una vez, usa en múltiples nodos

## Nodos incluidos

| Nodo | Descripción |
|------|-------------|
| **Load LLM Model (GGUF)** | Carga un modelo GGUF en memoria para reutilizar |
| **Prompt Translator & Enhancer (Multi→EN)** | Todo-en-uno: carga modelo, traduce y mejora |
| **Prompt Translator & Enhancer From Model (Multi→EN)** | Usa un modelo ya cargado para traducir y mejorar |

## Instalación

1. Clonar o copiar este directorio en `ComfyUI/custom_nodes/ComfyUI-PromptEnhancer/`
2. Instalar dependencias:
   ```bash
   pip install llama-cpp-python
   ```
   Para soporte GPU (CUDA):
   ```bash
   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```
3. Colocar modelos GGUF en `ComfyUI/models/LLM/` (subdirectorios permitidos)

## Modelos recomendados

> **IMPORTANTE**: Recomendamos **fuertemente** usar modelos **Instruct** o **Chat** (que suelen tener "Instruct" o "Chat" en su nombre, ej. `Qwen3-VL-4B-Instruct-Q4.gguf`) en lugar de modelos base. Los modelos Instruct siguen nuestras reglas mucho mejor y no requieren del *fallback* de completado raw que es más lento.
> *Nota: Los idiomas de entrada que el nodo puede entender dependen de si el modelo GGUF fue entrenado en esos idiomas.*

- **Qwen3-4B-Instruct-Q4_K_M.gguf** (~2.5 GB) — Buen balance velocidad/calidad, excelente soporte multi-idioma.
- **Qwen2.5-3B-Instruct-Q4_K_M.gguf** (~2 GB) — Más ligero.
- **Phi-3-mini-4k-instruct-Q4_K_M.gguf** (~2.3 GB) — Alternativa sólida.

## Uso

### Opción 1: Todo-en-uno
```
[Prompt Translator & Enhancer (Multi→EN)] → [CLIP Text Encode] → [KSampler]
```

### Opción 2: Modelo reutilizable
```
[Load LLM Model] → [Prompt Translator & Enhancer From Model (Multi→EN)] → [CLIP Text Encode] → [KSampler]
```

### Niveles de mejora

- **basic**: Traduce y añade tags mínimos de calidad
- **detailed**: Traduce y añade iluminación, composición, calidad
- **creative**: Traduce con interpretación artística, estilos, efectos
