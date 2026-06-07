def enable_attention_prefill_prefix(model_name, model):
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        from .ape_llama import enable_llama_attention_prefill_prefix
        enable_llama_attention_prefill_prefix(model)
    elif "mistral" in model_name_lower:
        from .ape_mistral import enable_mistral_attention_prefill_prefix
        enable_mistral_attention_prefill_prefix(model)
    elif "gemma" in model_name_lower:
        from .ape_gemma import enable_gemma_attention_prefill_prefix
        enable_gemma_attention_prefill_prefix(model)
    elif "qwen" in model_name_lower:
        from .ape_qwen import enable_qwen_attention_prefill_prefix
        enable_qwen_attention_prefill_prefix(model)
    else:
        raise ValueError(f"Unsupported APE model family: {model_name}")

def enable_attention_prefill_context(model_name, model):
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        from .ape_llama import enable_llama_attention_prefill_context
        enable_llama_attention_prefill_context(model)
    elif "mistral" in model_name_lower:
        from .ape_mistral import enable_mistral_attention_prefill_context
        enable_mistral_attention_prefill_context(model)
    elif "gemma" in model_name_lower:
        from .ape_gemma import enable_gemma_attention_prefill_context
        enable_gemma_attention_prefill_context(model)
    elif "qwen" in model_name_lower:
        from .ape_qwen import enable_qwen_attention_prefill_context
        enable_qwen_attention_prefill_context(model)
    else:
        raise ValueError(f"Unsupported APE model family: {model_name}")

def enable_attention_prefill_query(model_name, model, temperature, scale, position_shift=0):
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        from .ape_llama import enable_llama_attention_prefill_query
        enable_llama_attention_prefill_query(model, temperature, scale, position_shift=position_shift)
    elif "mistral" in model_name_lower:
        from .ape_mistral import enable_mistral_attention_prefill_query
        enable_mistral_attention_prefill_query(model, temperature, scale, position_shift=position_shift)
    elif "gemma" in model_name_lower:
        from .ape_gemma import enable_gemma_attention_prefill_query
        enable_gemma_attention_prefill_query(model, temperature, scale, position_shift=position_shift)
    elif "qwen" in model_name_lower:
        from .ape_qwen import enable_qwen_attention_prefill_query
        enable_qwen_attention_prefill_query(model, temperature, scale, position_shift=position_shift)
    else:
        raise ValueError(f"Unsupported APE model family: {model_name}")
