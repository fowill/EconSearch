import os
from typing import Iterable, List

from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled via fallback
    OpenAI = None

PROVIDER_CONFIG = {
    "shubiaobiao": {
        "base_url": os.getenv("SHUBIAOBIAO_BASE_URL", "https://api.shubiaobiao.cn/v1/"),
        "model": os.getenv("SHUBIAOBIAO_MODEL", "gpt-4o-mini"),
    },
    "deepseek": {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
}

def _resolve_llm_config():
    provider = os.getenv("LLM_PROVIDER", "shubiaobiao").strip().lower()
    provider_cfg = PROVIDER_CONFIG.get(provider)
    if not provider_cfg:
        raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'. Configure PROVIDER_CONFIG or adjust .env.")

    base_url = os.getenv("OPENAI_BASE_URL") or provider_cfg["base_url"]
    model = os.getenv("OPENAI_MODEL") or provider_cfg["model"]

    if not base_url or not model:
        raise RuntimeError("LLM configuration incomplete. Check your .env settings.")

    return provider, base_url.rstrip("/") + "/", model


LLM_PROVIDER, BASE_URL, DEFAULT_MODEL = _resolve_llm_config()
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

_client = None


def _ensure_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")
    if OpenAI is None:
        raise RuntimeError("openai package not installed. See requirements.txt.")
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def _run_chat(
    messages: List[dict],
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str = DEFAULT_MODEL,
) -> str:
    client = _ensure_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_keywords(question: str, n_keywords: int = 6) -> List[str]:
    prompt = (
        "You act as an academic search assistant. Given the user's question, generate "
        f"{n_keywords} short English keyword phrases suitable for searching finance and economics papers. "
        "Return one keyword phrase per line. Avoid numbering or extra commentary."
    )
    try:
        text = _run_chat(
            [
                {"role": "system", "content": "You craft terse English keywords for literature search."},
                {"role": "user", "content": f"{prompt}\n\nUser question:\n{question}"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
    except Exception:
        return _fallback_keywords(question, n_keywords)

    keywords = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if not keywords:
        return _fallback_keywords(question, n_keywords)
    keywords = [kw.rstrip(".") for kw in keywords]
    return keywords[:n_keywords]


def _fallback_keywords(question: str, n_keywords: int) -> List[str]:
    tokens = [token.lower() for token in question.replace(",", " ").split() if len(token) > 2]
    if not tokens:
        return [question]
    unique_tokens = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
        if len(unique_tokens) >= n_keywords:
            break
    while len(unique_tokens) < n_keywords:
        unique_tokens.append(question)
    return unique_tokens[:n_keywords]


def answer_with_context(question: str, contexts: Iterable[str]) -> str:
    context_blocks = "\n\n".join(f"[Source {idx + 1}]\n{ctx.strip()}" for idx, ctx in enumerate(contexts) if ctx.strip())
    if not context_blocks:
        return "No relevant documents were found for the question."

    try:
        return _run_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an academic assistant. Provide every response bilingually: "
                        "first write the complete English answer, then provide a faithful Simplified Chinese translation. "
                        "Preserve structure (headings, bullet points) across both languages."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Use only the information in the sources to answer the user's question. Cite sources inline using [Source X].\n"
                        "Produce a bilingual response with the following structure (keep headings exactly as shown):\n\n"
                        "English:\nOverview: <one paragraph synthesizing what the retrieved papers collectively address>\nPapers:\n- [Source X Title] (Source X): <2-3 sentence summary highlighting contribution and evidence>\n- ...\n\n"
                        "Chinese:\n概述：<用中文概括这些文献共同讨论的问题>\n文献：\n- [Source X 标题]（Source X）：<用中文概述该文献的核心贡献>\n- ...\n\n"
                        "Ensure the Chinese section is a faithful, fluent translation of the English section (not word-for-word).\n\n"
                        f"Sources:\n{context_blocks}\n\nQuestion:\n{question}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=800,
        )
    except Exception as exc:
        return f"(LLM unavailable) Unable to answer due to: {exc}"


def summarize_document(title: str, text: str, max_tokens: int = 700) -> str:
    """Produce a concise summary for an entire document."""
    cleaned = text.strip()
    if not cleaned:
        return "No content available to summarize."
    prompt = (
        "You are an expert academic summarizer. Produce a concise, structured summary for the paper below. "
        "First explain in one paragraph what overarching problem or question the paper tackles, then briefly cover its approach and findings. "
        "Write the summary bilingually with this structure (keep headings exact):\n\n"
        "English:\nOverview: <what topic/question the paper investigates>\nHighlights:\n- Method & Data: <one bullet>\n- Key Findings: <one bullet>\n- Limitations: <optional bullet if present>\n\n"
        "Chinese:\n概述：<论文关注的核心问题>\n要点：\n- 方法与数据：<一条中文摘要>\n- 主要结论：<一条中文摘要>\n- 局限性：<如有则简述>\n\n"
        "Ensure the Chinese section is a fluent translation of the English section.\n\n"
        f"Title: {title}\n\n"
        f"Full Text:\n{cleaned}"
    )
    try:
        return _run_chat(
            [
                {"role": "system", "content": "You summarize academic papers clearly and accurately."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return f"(LLM unavailable) Unable to summarize due to: {exc}"


__all__ = ["generate_keywords", "answer_with_context", "summarize_document"]
