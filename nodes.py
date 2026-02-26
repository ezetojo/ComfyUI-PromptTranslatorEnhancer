"""
ComfyUI-PromptTranslator-Enhancer
Custom nodes that enhance and translate prompts from Spanish to English
using a local GGUF model via llama-cpp-python.
"""

import os
import glob
import folder_paths


def _get_llama_class():
    """Lazy import of Llama to avoid errors when llama-cpp-python is not installed."""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed. Please install it with:\n"
            "  pip install llama-cpp-python\n"
            "For GPU (CUDA) support:\n"
            "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall --no-cache-dir"
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Register LLM model folder so ComfyUI knows about it
LLM_DIR = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(LLM_DIR):
    os.makedirs(LLM_DIR, exist_ok=True)


def get_available_gguf_models():
    """Recursively find all .gguf files under models/LLM/, excluding mmproj files."""
    pattern = os.path.join(LLM_DIR, "**", "*.gguf")
    all_files = glob.glob(pattern, recursive=True)
    # Filter out vision projector files (mmproj-*) and return relative paths
    models = []
    for f in all_files:
        basename = os.path.basename(f)
        if basename.lower().startswith("mmproj-"):
            continue
        rel_path = os.path.relpath(f, LLM_DIR)
        models.append(rel_path)
    return sorted(models) if models else ["no_models_found"]


# System prompts per enhancement level
SYSTEM_PROMPTS = {
    "basic": (
        "You are an AI assistant. Translate the following user prompt directly into English for an image generator. "
        "Do not add quotes, do not say 'Here is the translation'. Just output the translated English tags."
    ),
    "detailed": (
        "You are an expert prompt engineer. You will receive an image generation prompt in any language. "
        "Your task is to translate it to English and enhance it with rich, descriptive details "
        "(lighting, camera, composition, high quality). Output ONLY a comma-separated list of tags in English. "
        "Never repeat tags. Do not add conversational text."
    ),
    "creative": (
        "You are a creative art director. You will receive an image generation prompt in any language. "
        "Your task is to translate it to English and transform it into a stunning, highly artistic, and uniquely styled prompt. "
        "Output ONLY a comma-separated list of English tags. Never repeat tags. Do not add conversational text."
    ),
}



def deduplicate_tags(prompt):
    """Aggressive post-process to remove duplicates and stop long repetitions."""
    if not prompt:
        return prompt
    # Remove obvious AI filler like "Output:" or "English:" if model fails instructions
    for prefix in ["Output:", "Enhanced:", "English:", "Prompt:"]:
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].strip()

    tags = [t.strip() for t in prompt.split(",")]
    seen = set()
    unique_tags = []
    for t in tags:
        t_lower = t.lower().replace(".", "").replace("!", "").strip()
        # Detect catastrophic loop: if we start getting too many tags or same words
        if t_lower and t_lower not in seen and len(unique_tags) < 50:
            seen.add(t_lower)
            unique_tags.append(t)
    return ", ".join(unique_tags)


def run_llm_enhancement(llm_model, prompt_text, enhancement_level, max_tokens, temperature, seed):
    """Run the LLM to enhance and translate the prompt."""
    system_prompt = SYSTEM_PROMPTS.get(enhancement_level, SYSTEM_PROMPTS["detailed"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    # Attempt Instruct Chat Format first (best for Instruct/Chat models)
    try:
        response = llm_model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed if seed >= 0 else None,
            top_p=0.9,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[PromptEnhancer/Debug] Chat completion failed: {e}")
        result = ""

    # Fallback for Base models (non-instruct) that don't understand chat formats
    if not result:
        print("[PromptEnhancer/Debug] Empty response or base model detected. Falling back to raw text completion...")
        raw_prompt = f"{system_prompt}\nInput: {prompt_text}\nOutput:"
        try:
            raw_response = llm_model.create_completion(
                prompt=raw_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed if seed >= 0 else None,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\n\n", "Input:", "Output:"],
            )
            result = raw_response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[PromptEnhancer/Debug] Raw completion also failed: {e}")
            result = ""

    print(f"[PromptEnhancer/Debug] Raw LLM Reply: {repr(result)}")

    # Clean up: remove thinking blocks if model outputs them (Qwen3 /think)
    if "<think>" in result and "</think>" in result:
        think_end = result.rfind("</think>")
        result = result[think_end + len("</think>"):].strip()
    elif "<think>" in result:
        # Incomplete think block, remove everything from <think> onwards
        result = result[:result.find("<think>")].strip()

    # Remove surrounding quotes if present
    if (result.startswith('"') and result.endswith('"')) or \
       (result.startswith("'") and result.endswith("'")):
        result = result[1:-1].strip()

    return deduplicate_tags(result)


# ---------------------------------------------------------------------------
# Node: Load LLM Model
# ---------------------------------------------------------------------------

class LoadLLMModel:
    """Loads a GGUF language model into memory for reuse across multiple nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (get_available_gguf_models(), {"default": get_available_gguf_models()[0]}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1,
                                         "tooltip": "Number of layers to offload to GPU. -1 = all layers."}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 256,
                                   "tooltip": "Context window size in tokens. Qwen3 supports up to 40960."}),
            }
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "load_model"
    CATEGORY = "LLM/Prompt Enhancement"
    DESCRIPTION = "Loads a GGUF language model for prompt enhancement. Place .gguf files in ComfyUI/models/LLM/"

    def load_model(self, llm_model, n_gpu_layers, n_ctx):
        model_path = os.path.join(LLM_DIR, llm_model)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[PromptTranslator-Enhancer] Loading LLM: {llm_model}")
        print(f"[PromptTranslator-Enhancer]   GPU layers: {n_gpu_layers}, Context: {n_ctx}")

        Llama = _get_llama_class()
        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        print(f"[PromptTranslator-Enhancer] Model loaded successfully!")
        return (model,)


# ---------------------------------------------------------------------------
# Node: Prompt Enhancer & Translator (standalone - loads its own model)
# ---------------------------------------------------------------------------

class PromptEnhancerTranslator:
    """Takes a Spanish prompt, enhances it for image generation, and translates to English.
    This node loads the model on each execution. For repeated use, prefer LoadLLMModel + PromptEnhancerFromModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Escribe tu prompt en español..."}),
                "llm_model": (get_available_gguf_models(), {"default": get_available_gguf_models()[0]}),
                "enhancement_level": (["basic", "detailed", "creative"], {"default": "detailed"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 1024, "step": 32}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("enhanced_prompt", "original_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "LLM/Prompt Enhancement"
    DESCRIPTION = "Enhances and translates a Spanish prompt to English using a local GGUF model."

    def enhance(self, prompt, llm_model, enhancement_level, n_gpu_layers, max_tokens, temperature, seed):
        if not prompt.strip():
            return ("", prompt)

        model_path = os.path.join(LLM_DIR, llm_model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[PromptTranslator-Enhancer] Loading model: {llm_model}")
        Llama = _get_llama_class()
        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            verbose=False,
        )

        enhanced = run_llm_enhancement(model, prompt, enhancement_level, max_tokens, temperature, seed)

        # Free model memory
        del model

        print(f"[PromptTranslator-Enhancer] Original: {prompt}")
        print(f"[PromptTranslator-Enhancer] Enhanced: {enhanced}")

        return (enhanced, prompt)


# ---------------------------------------------------------------------------
# Node: Prompt Enhancer From Model (uses pre-loaded model)
# ---------------------------------------------------------------------------

class PromptEnhancerFromModel:
    """Takes a Spanish prompt and a pre-loaded LLM model, enhances the prompt and translates to English."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Escribe tu prompt en español..."}),
                "enhancement_level": (["basic", "detailed", "creative"], {"default": "detailed"}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 1024, "step": 32}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("enhanced_prompt", "original_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "LLM/Prompt Enhancement"
    DESCRIPTION = "Enhances and translates a Spanish prompt using a pre-loaded LLM model from LoadLLMModel node."

    def enhance(self, llm_model, prompt, enhancement_level, max_tokens, temperature, seed):
        if not prompt.strip():
            return ("", prompt)

        enhanced = run_llm_enhancement(llm_model, prompt, enhancement_level, max_tokens, temperature, seed)

        print(f"[PromptTranslator-Enhancer] Original: {prompt}")
        print(f"[PromptTranslator-Enhancer] Enhanced: {enhanced}")

        return (enhanced, prompt)


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LoadLLMModel": LoadLLMModel,
    "PromptEnhancerTranslator": PromptEnhancerTranslator,
    "PromptEnhancerFromModel": PromptEnhancerFromModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLLMModel": "Load LLM Model (GGUF)",
    "PromptEnhancerTranslator": "Prompt Translator & Enhancer (Multi→EN)",
    "PromptEnhancerFromModel": "Prompt Translator & Enhancer From Model (Multi→EN)",
}
