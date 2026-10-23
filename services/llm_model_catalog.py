"""获取供应商可用模型列表。

思路参考 cc-switch 的 model_fetch：
- 兼容 OpenAI 风格的 GET /models，按候选 URL 依次尝试，404/405 才换下一个候选；
- base URL 已带版本段（/v1、智谱 /paas/v4）时拼 {base}/models，否则拼 {base}/v1/models；
- base URL 命中 Anthropic 等兼容子路径时，追加剥离子路径后的兜底候选；
- 响应兼容 data[].id 与 models[].slug / id / name 两种格式。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 15.0

# 错误响应体可能是一整页 HTML，截断后再回传
ERROR_BODY_MAX_CHARS = 512

# 已知的协议兼容子路径后缀，按长度降序，保证最长前缀优先命中
KNOWN_COMPAT_SUFFIXES = (
    "/api/claudecode",
    "/api/anthropic",
    "/apps/anthropic",
    "/api/coding",
    "/claudecode",
    "/anthropic",
    "/step_plan",
    "/coding",
    "/claude",
)


class ModelFetchError(Exception):
    """模型列表获取失败，消息已脱敏，可直接返回给前端。"""


def _ends_with_version_segment(url: str) -> bool:
    """base URL 是否以 /v{N} 结尾，例如 /v1、/api/coding/paas/v4。"""
    last = url.rstrip("/").rsplit("/", 1)[-1]
    if not last.startswith("v"):
        return False
    digits = last[1:]
    return bool(digits) and digits.isdigit()


def _strip_compat_suffix(base_url: str) -> str | None:
    for suffix in KNOWN_COMPAT_SUFFIXES:
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return None


def build_models_url_candidates(
    base_url: str,
    models_url: str | None = None,
    is_full_url: bool = False,
) -> list[str]:
    """构造模型列表端点的候选 URL，按优先级排序并去重。"""
    if models_url and models_url.strip():
        return [models_url.strip()]

    trimmed = (base_url or "").strip().rstrip("/")
    if not trimmed:
        raise ModelFetchError("API Base URL 为空")

    candidates: list[str] = []

    if is_full_url:
        idx = trimmed.find("/v1/")
        if idx != -1:
            candidates.append(f"{trimmed[:idx]}/v1/models")
        else:
            slash = trimmed.rfind("/")
            root = trimmed[:slash] if slash != -1 else ""
            if "://" in root and len(root) > root.index("://") + 3:
                candidates.append(f"{root}/v1/models")
        if not candidates:
            raise ModelFetchError("无法从完整 URL 推导出模型端点")
        return candidates

    if _ends_with_version_segment(trimmed):
        candidates.append(f"{trimmed}/models")
        # 版本段不是 /v1 时（如 /v4），保留 /v1/models 作为兜底
        if not trimmed.endswith("/v1"):
            candidates.append(f"{trimmed}/v1/models")
    else:
        candidates.append(f"{trimmed}/v1/models")

    stripped = _strip_compat_suffix(trimmed)
    if stripped:
        root = stripped.rstrip("/")
        if root and "://" in root:
            candidates.append(f"{root}/v1/models")
            candidates.append(f"{root}/models")

    unique: list[str] = []
    for url in candidates:
        if url not in unique:
            unique.append(url)
    return unique


def _catalog_ids(models: Any) -> list[str]:
    """models[] 条目：slug 优先，回退 id，再回退 name 的 models/ 前缀。"""
    if not isinstance(models, list):
        return []
    ids: list[str] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        value = entry.get("slug") or entry.get("id")
        if not isinstance(value, str) or not value:
            name = entry.get("name")
            value = name[len("models/"):] if isinstance(name, str) and name.startswith("models/") else ""
        if value:
            ids.append(value)
    return ids


def parse_models_payload(payload: Any) -> list[dict]:
    """解析 /models 响应，返回 [{id, owned_by}]。"""
    if not isinstance(payload, dict):
        raise ModelFetchError("模型列表响应格式无法识别")

    models: list[dict] = []
    data = payload.get("data")
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id:
                models.append({"id": model_id, "owned_by": entry.get("owned_by")})
    elif payload.get("models") is not None:
        models = [{"id": mid, "owned_by": None} for mid in _catalog_ids(payload.get("models"))]

    if not models:
        raise ModelFetchError("响应中没有可用的模型条目")

    models.sort(key=lambda m: m["id"])
    return models


def _truncate(body: str) -> str:
    return body if len(body) <= ERROR_BODY_MAX_CHARS else body[:ERROR_BODY_MAX_CHARS] + "…"


async def fetch_models(
    api_key: str,
    base_url: str,
    models_url: str | None = None,
    is_full_url: bool = False,
) -> dict:
    """请求供应商的模型列表端点，返回 {models, endpoint}。"""
    if not api_key:
        raise ModelFetchError("缺少 API Key")

    candidates = build_models_url_candidates(base_url, models_url, is_full_url)
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    last_error = "没有可用的候选端点"

    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for url in candidates:
            try:
                response = await client.get(url, headers=headers)
            except httpx.HTTPError as e:
                raise ModelFetchError(f"请求失败: {type(e).__name__}") from e

            if response.is_success:
                try:
                    payload = response.json()
                except ValueError as e:
                    raise ModelFetchError("响应不是合法的 JSON") from e
                models = parse_models_payload(payload)
                logger.info(f"[ModelFetch] {url} 返回 {len(models)} 个模型")
                return {"models": models, "endpoint": url}

            body = _truncate(response.text.replace(api_key, "***"))
            if response.status_code in (404, 405):
                last_error = f"HTTP {response.status_code}: {body}"
                continue
            raise ModelFetchError(f"HTTP {response.status_code}: {body}")

    raise ModelFetchError(f"所有候选端点均失败: {last_error}")
