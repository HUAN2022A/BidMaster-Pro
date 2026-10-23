from __future__ import annotations

import logging

from core.llm_gateway.gateway import LLMGateway
from core.settings import get_settings

logger = logging.getLogger(__name__)

_gateway: LLMGateway | None = None

# llm_provider_configs 中 is_default=True 的配置，优先于 .env
_db_config: dict | None = None


def _fallback_models() -> list[str]:
    settings = get_settings()
    return [m.strip() for m in settings.llm_fallback_modes.split(",") if m.strip()]


def get_llm_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        settings = get_settings()
        if _db_config:
            provider = {
                "provider_id": _db_config["provider_id"],
                "api_key": _db_config["api_key"],
                "api_base": _db_config["api_base"],
            }
            default_model = _db_config["default_model"]
        else:
            provider = {
                "provider_id": "",
                "api_key": settings.llm_api_key,
                "api_base": settings.llm_api_base,
            }
            default_model = settings.llm_default_model
        _gateway = LLMGateway({
            "provider_id": provider["provider_id"],
            "providers": [
                {
                    "api_key": provider["api_key"],
                    "api_base": provider["api_base"],
                }
            ],
            "default_model": default_model,
            "fallback_models": _fallback_models(),
            "max_retries": settings.llm_max_retries,
        })
    return _gateway


def reset_llm_gateway():
    global _gateway
    _gateway = None


async def load_llm_runtime_config(db=None) -> None:
    """从 llm_provider_configs 载入默认配置，没有可用配置时回落到 .env。

    传入 db 时复用调用方的会话（配置刚写入、尚未提交也能读到）。
    """
    global _db_config
    from sqlalchemy import select

    from services.models import LLMProviderConfig

    owns_session = False
    if db is None:
        from services.database import async_session, is_db_ready

        if not is_db_ready():
            _db_config = None
            reset_llm_gateway()
            return
        db = async_session()()
        owns_session = True

    try:
        result = await db.execute(
            select(LLMProviderConfig)
            .where(
                LLMProviderConfig.is_default == True,  # noqa: E712
                LLMProviderConfig.enabled == True,  # noqa: E712
            )
            .order_by(LLMProviderConfig.created_at)
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row and row.api_key and row.default_model:
            _db_config = {
                "provider_id": row.provider_id,
                "api_key": row.api_key,
                "api_base": row.api_base or "",
                "default_model": row.default_model,
            }
            logger.info(
                f"LLM 运行时配置来自数据库: provider={row.provider_id}, model={row.default_model}"
            )
        else:
            if row:
                logger.warning("默认 LLM 配置缺少 api_key 或 default_model，已忽略，使用 .env")
            else:
                logger.info("数据库无默认 LLM 配置，运行时使用 .env")
            _db_config = None
    except Exception as e:
        logger.warning(f"加载 LLM 数据库配置失败，使用 .env: {e}")
        _db_config = None
    finally:
        if owns_session:
            await db.close()

    reset_llm_gateway()
