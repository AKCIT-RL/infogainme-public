"""Token counting utilities for LLM responses."""

from __future__ import annotations

import re
from typing import Optional, Tuple, Any


# Cache de tokenizers para evitar recarregar
_tokenizer_cache: dict[str, Any] = {}


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Conta tokens em um texto usando tokenizer específico do modelo.
    
    Suporta Llama 3.1 e Qwen 3 com tokenizers do transformers.
    Fallback para aproximação simples se tokenizer não disponível.
    
    Args:
        text: Texto para contar tokens.
        model: Nome do modelo (para escolher tokenizer correto).
        
    Returns:
        Número de tokens.
    """
    if not text:
        return 0
    
    if not model:
        # Fallback: aproximação simples (1 token ≈ 4 caracteres)
        return len(text) // 4
    
    model_lower = model.lower()
    
    # Tentar usar transformers para modelos específicos
    try:
        from transformers import AutoTokenizer
        
        # Detectar modelo e carregar tokenizer apropriado (com cache)
        tokenizer_key = None
        if "llama" in model_lower and "3.1" in model_lower:
            tokenizer_key = "llama-3.1"
            if tokenizer_key not in _tokenizer_cache:
                _tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
            tokenizer = _tokenizer_cache[tokenizer_key]
        elif "qwen" in model_lower and "3" in model_lower:
            tokenizer_key = "qwen-3"
            if tokenizer_key not in _tokenizer_cache:
                _tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
            tokenizer = _tokenizer_cache[tokenizer_key]
        else:
            # Modelo não reconhecido, usar aproximação
            return len(text) // 4
        
        # Contar tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
        
    except (ImportError, Exception):
        # Fallback: aproximação simples (1 token ≈ 4 caracteres)
        return len(text) // 4


def extract_reasoning_and_final(raw_content: str) -> Tuple[str, str]:
    """Extrai reasoning e resposta final de conteúdo raw.
    
    Procura por tags <think> ou similares.
    Também trata casos onde há apenas </think> sem tag de abertura.
    
    Args:
        raw_content: Conteúdo raw da resposta do LLM.
        
    Returns:
        Tupla (reasoning_text, final_text).
    """
    if not raw_content:
        return "", ""
    
    # Padrões de reasoning tags (ordem importa: mais específicos primeiro)
    # Usa os mesmos padrões que llm_final_content em utils.py
    reasoning_patterns = [
        r'<think>(.*?)</think>',
        r'<think>(.*?)</think>',
        r'<reasoning>(.*?)</reasoning>',
        r'<thinking>(.*?)</thinking>',
    ]
    
    reasoning_text = ""
    final_text = raw_content
    
    # Caso especial: apenas tag de fechamento </think> sem abertura
    # (usado por alguns modelos como Qwen3-30B-A3B-Thinking-2507)
    # Verificar isso ANTES dos padrões completos para evitar falsos positivos
    if '</think>' in raw_content and '<think>' not in raw_content:
        # Procurar por </think> e extrair tudo antes dela como reasoning
        closing_tag_pattern = r'</think>'
        match = re.search(closing_tag_pattern, raw_content, re.IGNORECASE)
        if match:
            # Tudo antes da tag de fechamento é reasoning
            reasoning_text = raw_content[:match.start()].strip()
            # Tudo depois da tag de fechamento é a resposta final
            final_text = raw_content[match.end():].strip()
    
    # Tentar extrair reasoning de cada padrão (apenas se não encontrou no caso especial)
    if not reasoning_text:
        for pattern in reasoning_patterns:
            matches = re.finditer(pattern, raw_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                reasoning_text += match.group(1) + "\n"
                # Remover reasoning do texto final
                final_text = final_text.replace(match.group(0), "")
    
    # Se não encontrou tags fechadas, tentar padrão unclosed
    if not reasoning_text:
        unclosed_patterns = [
            r'<think>(.*)',
            r'<think>(.*)',
            r'<reasoning>(.*)',
        ]
        for pattern in unclosed_patterns:
            match = re.search(pattern, raw_content, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning_text = match.group(1)
                final_text = raw_content[:match.start()] + raw_content[match.end():]
                break
    
    # Limpar whitespace excessivo
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    reasoning_text = reasoning_text.strip()
    
    return reasoning_text, final_text


def count_seeker_tokens(
    reasoning_history: list[dict],
    history: list[dict],
    model: Optional[str] = None
) -> Tuple[int, Optional[int], int]:
    """Conta tokens do SeekerAgent separando reasoning e resposta final.
    
    Args:
        reasoning_history: Lista de mensagens com reasoning (raw).
        history: Lista de mensagens limpas (sem reasoning).
        model: Nome do modelo para contagem precisa.
        
    Returns:
        Tupla (total_tokens, reasoning_tokens, final_tokens).
        reasoning_tokens será None se não houver reasoning.
    """
    total_tokens = 0
    reasoning_tokens = 0
    final_tokens = 0
    has_reasoning = False
    
    # Usar reasoning_history se disponível e não vazio
    if reasoning_history:
        for msg in reasoning_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    reasoning_text, final_text = extract_reasoning_and_final(content)
                    
                    if reasoning_text:
                        has_reasoning = True
                        r_tokens = count_tokens(reasoning_text, model)
                        f_tokens = count_tokens(final_text, model)
                        reasoning_tokens += r_tokens
                        final_tokens += f_tokens
                        total_tokens += r_tokens + f_tokens
                    else:
                        # Sem reasoning, contar apenas final
                        f_tokens = count_tokens(content, model)
                        final_tokens += f_tokens
                        total_tokens += f_tokens
    
    # Se não há reasoning_history ou está vazio, usar history
    if not has_reasoning and history:
        for msg in history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    tokens = count_tokens(content, model)
                    final_tokens += tokens
                    total_tokens += tokens
    
    return total_tokens, reasoning_tokens if has_reasoning else None, final_tokens

