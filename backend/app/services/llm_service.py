"""LLM 服务封装。

这里统一管理：
1. ChatOpenAI 兼容模型实例的创建与缓存。
2. 文本 / JSON 输出的异步调用辅助函数。
3. LangSmith Trace 的 run_name / tags 透传。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import get_effective_llm_config

logger = logging.getLogger(__name__)

_llm_cache: dict[Tuple[str, str, str, float, float, int, int], BaseChatModel] = {}
_llm_extra_retries = max(0, int(os.getenv("LLM_EXTRA_RETRIES", "1")))
_llm_light_run_names = {"supervisor", "attraction_agent", "weather_agent", "hotel_agent"}
_llm_heavy_run_names = {"planner_agent"}
_llm_light_semaphore = asyncio.Semaphore(
    max(1, int(os.getenv("LLM_LIGHT_MAX_CONCURRENCY", os.getenv("LLM_MAX_CONCURRENCY", "3"))))
)
_llm_heavy_semaphore = asyncio.Semaphore(
    max(1, int(os.getenv("LLM_HEAVY_MAX_CONCURRENCY", "1")))
)


def _make_signature(
    llm_config: dict[str, Any],
    *,
    temperature: float,
    timeout: float,
    max_tokens: int,
) -> Tuple[str, str, str, float, float, int, int]:
    """构造缓存签名，保证不同 agent 可以按各自参数复用模型实例。"""
    return (
        str(llm_config["api_key"]),
        str(llm_config["base_url"]),
        str(llm_config["model"]),
        float(temperature),
        float(timeout),
        int(llm_config["max_retries"]),
        int(max_tokens),
    )


def get_llm(
    *,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    max_tokens: int = 2000,
) -> BaseChatModel:
    """获取 ChatOpenAI 实例。

    为了兼顾不同 agent 的风格：
    - supervisor / planner 一般使用低温度模型；
    - 如果后续需要更保守或更发散的 agent，可以直接传入不同 temperature。
    """
    llm_config = get_effective_llm_config()
    resolved_temperature = float(temperature if temperature is not None else llm_config["temperature"])
    resolved_timeout = float(timeout if timeout is not None else llm_config["timeout"])

    api_key = llm_config["api_key"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY/LLM_API_KEY 未配置")

    signature = _make_signature(
        llm_config,
        temperature=resolved_temperature,
        timeout=resolved_timeout,
        max_tokens=max_tokens,
    )
    if signature not in _llm_cache:
        _llm_cache[signature] = ChatOpenAI(
            api_key=api_key,
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            temperature=resolved_temperature,
            max_tokens=max_tokens,
            timeout=resolved_timeout,
            max_retries=int(llm_config["max_retries"]),
        )
        logger.info(
            "初始化 LLM 实例: model=%s temperature=%.2f timeout=%.1fs max_tokens=%d",
            llm_config["model"],
            resolved_temperature,
            resolved_timeout,
            max_tokens,
        )

    return _llm_cache[signature]


async def ainvoke_text(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    timeout: Optional[float] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """异步调用模型并返回文本。"""
    llm = get_llm(temperature=temperature, timeout=timeout, max_tokens=max_tokens)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    config: dict[str, Any] = {}
    if run_name:
        config["run_name"] = run_name
    if tags:
        config["tags"] = tags

    last_error: Exception | None = None
    for attempt in range(_llm_extra_retries + 1):
        try:
            async with _pick_llm_semaphore(run_name):
                response = await llm.ainvoke(messages, config=config or None)
            break
        except Exception as exc:
            last_error = exc
            if attempt >= _llm_extra_retries or not _should_retry_llm_error(exc):
                raise
            logger.warning(
                "LLM 调用失败，准备重试: run_name=%s attempt=%d error=%s",
                run_name or "-",
                attempt + 1,
                exc.__class__.__name__,
            )
            await asyncio.sleep(min(1.5, 0.4 * (attempt + 1)))
    else:
        assert last_error is not None
        raise last_error

    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


async def ainvoke_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    timeout: Optional[float] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> tuple[Any, str]:
    """异步调用模型并解析 JSON。"""
    raw_text = await ainvoke_text(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        run_name=run_name,
        tags=tags,
    )
    return extract_json_payload(raw_text), raw_text


def extract_json_payload(text: str) -> Any:
    """从模型返回文本中提取 JSON。

    兼容三种常见格式：
    1. 纯 JSON。
    2. ```json fenced code block。
    3. 文本里夹带一个 JSON 对象 / 数组。
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("模型未返回可解析内容")

    fenced_match = re.search(r"```json\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1).strip())

    if raw.startswith("{") or raw.startswith("["):
        return json.loads(raw)

    object_match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if object_match:
        return json.loads(object_match.group(1).strip())

    raise ValueError(f"无法从模型输出中提取 JSON: {raw[:200]}")


def _should_retry_llm_error(exc: Exception) -> bool:
    """判断当前异常是否适合在应用层重试。"""
    text = f"{exc.__class__.__name__}: {exc}".upper()
    retry_terms = (
        "TIMEOUT",
        "TIMED OUT",
        "READTIMEOUT",
        "APITIMEOUTERROR",
        "APICONNECTIONERROR",
        "CONNECTION",
        "502",
        "503",
        "504",
    )
    return any(term in text for term in retry_terms)


def _pick_llm_semaphore(run_name: Optional[str]) -> asyncio.Semaphore:
    """按任务类型选择 LLM 并发池。

    轻量检索类 agent 允许并发占用远端 LLM，
    planner 这类重任务默认单独串行，避免多个大响应互相拖慢。
    """
    normalized = (run_name or "").strip().lower()
    if normalized in _llm_heavy_run_names:
        return _llm_heavy_semaphore
    return _llm_light_semaphore


def reset_llm() -> None:
    """清空缓存的模型实例。"""
    _llm_cache.clear()
