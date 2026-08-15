import logging
from functools import lru_cache

from supabase import Client, create_client

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    if settings.environment.lower() in {"development", "dev", "local"}:
        logger.info(
            "Supabase client configuration: supabase_url_present=%s supabase_anon_key_present=%s",
            bool(settings.supabase_url.strip()),
            bool(settings.supabase_anon_key.strip()),
        )
    return create_client(settings.supabase_url, settings.supabase_anon_key)
