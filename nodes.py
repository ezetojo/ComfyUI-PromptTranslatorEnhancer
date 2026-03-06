"""
ComfyUI-PromptTranslator-Enhancer
Custom nodes that enhance and translate prompts from any language to English
using a local GGUF model via llama-cpp-python.
"""

import os
import sys
import glob
import ctypes
import folder_paths

# ---------------------------------------------------------------------------
# CUDA DLL Pre-loading  (critical for running inside ComfyUI on Windows)
# ---------------------------------------------------------------------------
_cuda_dlls_loaded = False


def _ensure_cuda_dlls():
    """Pre-register DLL directories so llama.cpp can find CUDA runtime libs.

    Inside ComfyUI, PyTorch loads its own copy of CUDA DLLs from
    ``site-packages/torch/lib``.  llama-cpp-python ships its own
    ``ggml-cuda.dll`` that depends on ``cudart64_*.dll`` and
    ``cublas64_*.dll`` from the *same* CUDA toolkit.  If those DLLs
    are not on PATH or registered via ``os.add_dll_directory``,
    ``llama_model_load_from_file`` silently returns NULL.

    This function registers every directory that might contain the
    needed CUDA runtime DLLs.
    """
    global _cuda_dlls_loaded
    if _cuda_dlls_loaded:
        return
    _cuda_dlls_loaded = True

    if sys.platform != "win32":
        return

    dll_dirs_to_register = set()

    # 1. llama_cpp's own bin/ and lib/ directories
    try:
        import llama_cpp as _lc
        pkg_dir = os.path.dirname(_lc.__file__)
        for subdir in ("bin", "lib"):
            d = os.path.join(pkg_dir, subdir)
            if os.path.isdir(d):
                dll_dirs_to_register.add(d)
    except ImportError:
        pass

    # 2. PyTorch's CUDA DLLs (cudart, cublas)
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            dll_dirs_to_register.add(torch_lib)
    except ImportError:
        pass

    # 3. NVIDIA CUDA toolkit on PATH (if installed)
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            dll_dirs_to_register.add(cuda_bin)

    # Register all directories
    for d in dll_dirs_to_register:
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
        # Also prepend to PATH as legacy fallback
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

    if dll_dirs_to_register:
        print(f"[PromptTranslator-Enhancer] Registered {len(dll_dirs_to_register)} CUDA DLL directories")


def _get_llama_class():
    """Lazy import of Llama to avoid errors when llama-cpp-python is not installed."""
    _ensure_cuda_dlls()
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed. Please install it with:\n"
            "  pip install llama-cpp-python\n"
            "For GPU (CUDA) support:\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir'
        )


# Vision-Language model filename patterns (these need special multimodal handlers)
_VL_PATTERNS = ("qwen3vl", "qwen2vl", "qwen2.5vl", "llava", "minicpm-v", "internvl")


def _is_vl_model(model_path):
    """Check if a GGUF file is a Vision-Language model (not suitable for text-only use)."""
    name_lower = os.path.basename(model_path).lower()
    # Also check parent folder name
    parent_lower = os.path.basename(os.path.dirname(model_path)).lower()
    combined = f"{parent_lower}/{name_lower}"
    return any(p in combined for p in _VL_PATTERNS)


def _free_torch_vram():
    """Ask PyTorch to release cached VRAM so llama.cpp can use it."""
    try:
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved(0) / 1024**2
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved(0) / 1024**2
            freed = before - after
            if freed > 1:
                print(f"[PromptTranslator-Enhancer]   Freed {freed:.0f} MiB of PyTorch VRAM cache")
    except Exception:
        pass


def _load_model_with_fallback(Llama, model_path, n_gpu_layers, n_ctx):
    """Try to load a GGUF model with an aggressive fallback strategy.

    Fallback order (tries freeing PyTorch VRAM cache before each):
      1. GPU  + requested n_ctx  (flash_attn=auto)
      2. GPU  + 2048             (flash_attn=auto)
      3. GPU  + 512              (flash_attn=auto)
      4. CPU  + requested n_ctx  (flash_attn=DISABLED, offload_kqv=False)
      5. CPU  + 2048             (flash_attn=DISABLED, offload_kqv=False)
      6. CPU  + 512              (flash_attn=DISABLED, offload_kqv=False)

    The CPU fallbacks explicitly disable Flash Attention and KQV offloading
    so that llama.cpp does NOT try to allocate CUDA compute buffers.

    Returns the loaded Llama model instance.
    Raises RuntimeError if all attempts fail.
    """
    # Warn about Vision-Language models
    if _is_vl_model(model_path):
        print(
            f"[PromptTranslator-Enhancer] ⚠ WARNING: '{os.path.basename(model_path)}' appears to be a "
            f"Vision-Language model (VL).  These models require a multimodal handler and are NOT "
            f"supported by this text-only node.  Please select a text/instruct model instead "
            f"(e.g. Qwen3-4B-Q4_K_M.gguf)."
        )
        raise RuntimeError(
            f"Model '{os.path.basename(model_path)}' is a Vision-Language model and cannot be used "
            f"with this text-only prompt enhancement node.  Please choose a regular text model."
        )

    # Flash Attention type values (llama.cpp enum)
    FLASH_ATTN_DISABLED = 0   # force off – no CUDA compute buffers

    # Build list of attempts: (label, kwargs_dict)
    ctx_sizes = sorted(set([n_ctx, 2048, 512]), reverse=True)  # dedupe & descending

    attempts = []
    # GPU attempts (flash_attn left at default/auto)
    if n_gpu_layers != 0:
        for ctx in ctx_sizes:
            attempts.append((f"GPU/ctx={ctx}", dict(
                n_gpu_layers=n_gpu_layers, n_ctx=ctx,
            )))
    # CPU attempts – disable flash_attn, kqv offload, AND op_offload
    # op_offload=False is critical: it prevents llama.cpp from allocating
    # ~606 MiB CUDA compute buffers even when the model is on CPU
    for ctx in ctx_sizes:
        attempts.append((f"CPU-pure/ctx={ctx}", dict(
            n_gpu_layers=0, n_ctx=ctx,
            flash_attn_type=FLASH_ATTN_DISABLED,
            offload_kqv=False,
            op_offload=False,
        )))

    last_error = None
    for label, kwargs in attempts:
        try:
            _free_torch_vram()
            gpu_layers = kwargs.get("n_gpu_layers", 0)
            ctx = kwargs.get("n_ctx", n_ctx)
            print(f"[PromptTranslator-Enhancer]   Trying {label} (n_gpu_layers={gpu_layers}) ...")
            model = Llama(
                model_path=model_path,
                verbose=True,   # ALWAYS verbose so native errors show in console
                **kwargs,
            )
            if ctx < n_ctx:
                print(f"[PromptTranslator-Enhancer]   ⚠ Context reduced from {n_ctx} to {ctx} to fit in memory")
            print(f"[PromptTranslator-Enhancer]   ✓ Model loaded successfully ({label})")
            return model
        except Exception as e:
            last_error = e
            print(f"[PromptTranslator-Enhancer]   ✗ {label} failed: {e}")

    raise RuntimeError(
        f"Failed to load model after {len(attempts)} attempts.\n"
        f"  Model: {model_path}\n"
        f"  Last error: {last_error}\n"
        f"  Troubleshooting:\n"
        f"    1. Make sure the .gguf file is a TEXT model (not Vision-Language)\n"
        f"    2. Make sure the .gguf file is valid and not corrupted\n"
        f"    3. Try freeing GPU VRAM by unloading other models first\n"
        f"    4. Check the console output above for native llama.cpp error messages"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Register LLM model folder so ComfyUI knows about it
LLM_DIR = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(LLM_DIR):
    os.makedirs(LLM_DIR, exist_ok=True)


def get_available_gguf_models():
    """Recursively find all .gguf files under models/LLM/, excluding mmproj and VL files."""
    pattern = os.path.join(LLM_DIR, "**", "*.gguf")
    all_files = glob.glob(pattern, recursive=True)
    # Filter out vision projector files (mmproj-*), VL models, and return relative paths
    models = []
    for f in all_files:
        basename = os.path.basename(f)
        if basename.lower().startswith("mmproj-"):
            continue
        if _is_vl_model(f):
            continue
        rel_path = os.path.relpath(f, LLM_DIR)
        models.append(rel_path)
    return sorted(models) if models else ["no_models_found"]


# System prompts per enhancement level
# Common suffix appended to all system prompts to suppress chain-of-thought
_NO_THINK = (
    " Do NOT include any reasoning, thinking, or explanation. "
    "Do NOT wrap your answer in <think> tags. "
    "Respond with ONLY the final comma-separated English tags, nothing else."
)

SYSTEM_PROMPTS = {
    "basic": (
        "You are an AI assistant. Translate the following user prompt directly into English for an image generator. "
        "Do not add quotes, do not say 'Here is the translation'. Just output the translated English tags."
        + _NO_THINK
    ),
    "detailed": (
        "You are an expert prompt engineer. You will receive an image generation prompt in any language. "
        "Your task is to translate it to English and enhance it with rich, descriptive details "
        "(lighting, camera, composition, high quality). Output ONLY a comma-separated list of tags in English. "
        "Never repeat tags. Do not add conversational text."
        + _NO_THINK
    ),
    "creative": (
        "You are a creative art director. You will receive an image generation prompt in any language. "
        "Your task is to translate it to English and transform it into a stunning, highly artistic, and uniquely styled prompt. "
        "Output ONLY a comma-separated list of English tags. Never repeat tags. Do not add conversational text."
        + _NO_THINK
    ),
}


import re

def _strip_thinking(text):
    """Remove chain-of-thought / reasoning blocks from LLM output.

    Handles:
      1. Explicit <think>...</think> blocks (complete or incomplete)
      2. Untagged reasoning text that Qwen3 sometimes emits after the answer
    """
    if not text:
        return text

    # 1. Remove complete <think>...</think> blocks (possibly multiple)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Remove incomplete <think> block (no closing tag) — keep text before it
    if '<think>' in text:
        text = text[:text.find('<think>')].strip()

    # 3. Remove </think> leftovers (e.g. response starts with </think>\n content)
    if '</think>' in text:
        text = text[text.rfind('</think>') + len('</think>'):].strip()

    # 4. Remove untagged reasoning that Qwen3 sometimes appends after the answer
    #    Strategy: scan lines and cut from the first "reasoning" line onwards
    _REASONING_STARTS = (
        'okay', 'wait', 'let me', 'the input', 'the original',
        'translating', 'translation', 'i think', "i'll", 'so i',
        'hmm', 'alright', 'so ', 'but ', 'however', 'note:',
        'the spanish', 'the difference', 'maybe', 'the user',
        'the provided', 'the prompt', 'the text', 'in the',
    )

    lines = text.split('\n')
    if len(lines) > 1:
        # Find the first line that looks like reasoning
        cut_at = None
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in _REASONING_STARTS):
                cut_at = i
                break
        if cut_at is not None and cut_at > 0:
            # Keep only lines before the reasoning started
            text = '\n'.join(lines[:cut_at]).strip()
        elif cut_at == 0:
            # The ENTIRE output is reasoning — try to find any comma-separated
            # tags line after the reasoning
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and ',' in stripped and not any(
                    stripped.lower().startswith(w) for w in _REASONING_STARTS
                ):
                    text = stripped
                    break

    return text


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
    result = _strip_thinking(result)

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
        model = _load_model_with_fallback(Llama, model_path, n_gpu_layers, n_ctx)

        return (model,)


# ---------------------------------------------------------------------------
# Node: Prompt Enhancer & Translator (standalone - loads its own model)
# ---------------------------------------------------------------------------

class PromptEnhancerTranslator:
    """Takes a prompt in any language, enhances it for image generation, and translates to English.
    This node loads the model on each execution. For repeated use, prefer LoadLLMModel + PromptEnhancerFromModel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Escribe tu prompt en cualquier idioma..."}),
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
    DESCRIPTION = "Enhances and translates a prompt to English using a local GGUF model."

    def enhance(self, prompt, llm_model, enhancement_level, n_gpu_layers, max_tokens, temperature, seed):
        if not prompt.strip():
            return ("", prompt)

        model_path = os.path.join(LLM_DIR, llm_model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[PromptTranslator-Enhancer] Loading model: {llm_model}")
        Llama = _get_llama_class()
        model = _load_model_with_fallback(Llama, model_path, n_gpu_layers, n_ctx=4096)

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
    """Takes a prompt in any language and a pre-loaded LLM model, enhances the prompt and translates to English."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                       "placeholder": "Escribe tu prompt en cualquier idioma..."}),
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
    DESCRIPTION = "Enhances and translates a prompt using a pre-loaded LLM model from LoadLLMModel node."

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
