from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.skill_engine.base import SkillContext
from services.database import get_db
from services.generate.skills.ai_image_skill import AiImageSkill
from services.llm_factory import get_llm_gateway
from services.middleware.api_key import (
    AuthPrincipal,
    require_any_auth,
    consume_credits,
    refund_credits,
)
from services.models import IMAGE_CONFIG_NOTE, LLMProviderConfig

logger = logging.getLogger(__name__)

# 路由级别：允许 API Key 或 Bearer Token 任一认证方式
router = APIRouter(dependencies=[Depends(require_any_auth)])

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

# AI 配图单次调用消耗的 credits 数
_IMG_CREDITS_COST = 5


class ImageGenerateRequest(BaseModel):
    prompt: str
    provider: str = "fallback"
    image_size: str = "landscape_16_9"
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    volcengine_api_key: str | None = None
    google_api_key: str | None = None


class ProviderConfig(BaseModel):
    volcengine_api_key: str | None = None
    google_api_key: str | None = None


class ImageProviderUpsert(BaseModel):
    provider_id: str
    display_name: str | None = None
    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None


@router.post("/generate")
async def generate_image(
    req: ImageGenerateRequest,
    principal: AuthPrincipal = Depends(require_any_auth),
    db: AsyncSession = Depends(get_db),
):
    """生成 AI 配图。

    - API Key (credits 类型)：扣 5 credits
    - API Key (subscription 类型)：禁止访问算力服务
    - Bearer Token：Web 端用户，无需扣 credits（按 RBAC 权限控制）
    """
    # API Key 类型校验：算力服务仅允许 credits 类型
    if principal.is_api_key:
        if principal.api_key.type != "credits":
            raise HTTPException(
                status_code=403,
                detail="AI 配图是算力服务，需要 credits 类型的 API Key",
            )
        if principal.api_key.credits_remaining < _IMG_CREDITS_COST:
            raise HTTPException(
                status_code=402,
                detail=f"Credits 余额不足：需要 {_IMG_CREDITS_COST}，剩余 {principal.api_key.credits_remaining}",
            )

    skill = AiImageSkill()
    gateway = get_llm_gateway()

    # 自定义供应商：请求未带齐时，复用平台设置里保存的同名 LLM 配置
    api_base = (req.api_base or "").strip()
    api_key = (req.api_key or "").strip()
    model = (req.model or "").strip()
    if req.provider not in ("default", "fallback", "volcengine", "google"):
        result = await db.execute(
            select(LLMProviderConfig)
            .where(
                LLMProviderConfig.provider_id == req.provider,
                LLMProviderConfig.enabled == True,  # noqa: E712
            )
            .order_by(LLMProviderConfig.created_at)
        )
        rows = result.scalars().all()
        # 优先使用配图配置，其次任意同名可用的 LLM 配置
        cfg = next((r for r in rows if r.note == IMAGE_CONFIG_NOTE), rows[0] if rows else None)
        if cfg:
            api_base = api_base or (cfg.api_base or "")
            api_key = api_key or cfg.api_key
            model = model or (cfg.default_model or "")

    parameters = {
        "prompt": req.prompt,
        "provider": req.provider,
        "image_size": req.image_size,
        "model": model,
        "api_base": api_base,
        "custom_api_key": api_key,
    }

    if req.volcengine_api_key:
        parameters["volcengine_api_key"] = req.volcengine_api_key
    elif req.provider == "volcengine":
        env_key = os.getenv("VOLCENGINE_API_KEY", "")
        if env_key:
            parameters["volcengine_api_key"] = env_key

    if req.google_api_key:
        parameters["google_api_key"] = req.google_api_key
    elif req.provider == "google":
        env_key = os.getenv("GOOGLE_API_KEY", "")
        if env_key:
            parameters["google_api_key"] = env_key

    ctx = SkillContext(
        project_id="",
        db=None,
        llm=gateway,
        parameters=parameters,
    )

    # 先扣费（避免白嫖）
    if principal.is_api_key and not principal.is_dev:
        await consume_credits(
            db, principal.api_key, _IMG_CREDITS_COST,
            endpoint="/api/ai-image/generate", method="POST",
        )

    try:
        result = await skill.safe_execute(ctx)

        if not result.success:
            # 失败退款
            if principal.is_api_key and not principal.is_dev:
                await refund_credits(
                    db, principal.api_key, _IMG_CREDITS_COST,
                    endpoint="/api/ai-image/generate",
                )
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "success": True,
            "data": result.data,
            "credits_cost": _IMG_CREDITS_COST if principal.is_api_key else 0,
            "credits_remaining": principal.api_key.credits_remaining if principal.is_api_key else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        # 异常退款
        if principal.is_api_key and not principal.is_dev:
            try:
                await refund_credits(
                    db, principal.api_key, _IMG_CREDITS_COST,
                    endpoint="/api/ai-image/generate",
                )
            except Exception as refund_err:
                logger.error(f"退款失败: {refund_err}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers")
async def list_providers(db: AsyncSession = Depends(get_db)):
    volcengine_key = os.getenv("VOLCENGINE_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    providers = [
        {
            "name": "volcengine",
            "display_name": "火山方舟",
            "configured": bool(volcengine_key),
            "description": "火山方舟视觉生成API",
        },
        {
            "name": "google",
            "display_name": "Google AI Studio (Imagen)",
            "configured": bool(google_key),
            "description": "Google Imagen 图片生成",
        },
        {
            "name": "fallback",
            "display_name": "默认服务",
            "configured": True,
            "description": "Trae text-to-image 免费服务",
        },
    ]

    # 合并平台设置中保存的 LLM 供应商配置，两边共用同一份数据
    result = await db.execute(
        select(LLMProviderConfig)
        .where(LLMProviderConfig.enabled == True)  # noqa: E712
        .order_by(LLMProviderConfig.created_at)
    )
    by_name = {p["name"]: p for p in providers}
    # 同一 provider_id 可能既有对话配置又有配图配置，这里按 provider_id 去重并优先取配图配置
    merged: dict[str, LLMProviderConfig] = {}
    for row in result.scalars().all():
        current = merged.get(row.provider_id)
        if current is None or (row.note == IMAGE_CONFIG_NOTE and current.note != IMAGE_CONFIG_NOTE):
            merged[row.provider_id] = row

    for row in merged.values():
        if row.provider_id in by_name:
            by_name[row.provider_id].update({
                "configured": bool(row.api_key),
                "model": row.default_model or "",
                "api_base": row.api_base or "",
                "is_image_config": row.note == IMAGE_CONFIG_NOTE,
                "llm_config_id": row.id,
            })
            continue
        providers.append({
            "name": row.provider_id,
            "display_name": row.display_name or row.provider_id,
            "configured": bool(row.api_key),
            "description": row.note or "平台 LLM 配置",
            "model": row.default_model or "",
            "api_base": row.api_base or "",
            "is_image_config": row.note == IMAGE_CONFIG_NOTE,
            "llm_config_id": row.id,
        })

    return {"providers": providers, "credits_cost_per_image": _IMG_CREDITS_COST}


@router.post("/providers")
async def save_image_provider(payload: ImageProviderUpsert, db: AsyncSession = Depends(get_db)):
    """保存配图供应商配置，写入平台设置的 LLM 供应商列表。"""
    import re

    if not re.match(r"^[A-Za-z0-9._\-]{1,64}$", payload.provider_id):
        raise HTTPException(status_code=400, detail="供应商 ID 格式不合法")
    api_base = (payload.api_base or "").strip()
    if api_base and not re.match(r"^https?://", api_base):
        raise HTTPException(status_code=400, detail="API Base URL 须以 http:// 或 https:// 开头")
    api_key = (payload.api_key or "").strip()
    if len(api_key) < 4:
        raise HTTPException(status_code=400, detail="API Key 不能为空")

    model = (payload.model or "").strip()
    result = await db.execute(
        select(LLMProviderConfig).where(
            LLMProviderConfig.provider_id == payload.provider_id,
            LLMProviderConfig.note == IMAGE_CONFIG_NOTE,
        )
    )
    cfg = result.scalars().first()
    if cfg:
        cfg.api_key = api_key
        cfg.api_base = api_base or cfg.api_base
        cfg.default_model = model or cfg.default_model
        cfg.display_name = payload.display_name or cfg.display_name
        cfg.enabled = True
    else:
        cfg = LLMProviderConfig(
            provider_id=payload.provider_id,
            display_name=payload.display_name,
            api_key=api_key,
            api_base=api_base,
            default_model=model or None,
            note=IMAGE_CONFIG_NOTE,
            enabled=True,
            is_default=False,
        )
        db.add(cfg)
    await db.flush()

    return {"success": True, "id": cfg.id}


@router.put("/config")
async def save_provider_config(config: ProviderConfig):
    lines: list[str] = []

    if _ENV_PATH.exists():
        with open(_ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

    keys_to_set: dict[str, str | None] = {
        "VOLCENGINE_API_KEY": config.volcengine_api_key,
        "GOOGLE_API_KEY": config.google_api_key,
    }

    updated_keys: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in keys_to_set.items():
            if stripped.startswith(f"{key}="):
                if value is not None:
                    lines[i] = f"{key}={value}\n"
                else:
                    lines[i] = f"{key}=\n"
                updated_keys.add(key)
                break

    for key, value in keys_to_set.items():
        if key not in updated_keys:
            lines.append(f"{key}={value or ''}\n")

    with open(_ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    if config.volcengine_api_key:
        os.environ["VOLCENGINE_API_KEY"] = config.volcengine_api_key
    if config.google_api_key:
        os.environ["GOOGLE_API_KEY"] = config.google_api_key

    return {
        "success": True,
        "message": "API配置已保存",
        "volcengine_configured": bool(config.volcengine_api_key),
        "google_configured": bool(config.google_api_key),
    }
