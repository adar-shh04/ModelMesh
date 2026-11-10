"""Operator command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from modelmesh.auth.admin import AdminService, KeyDefaults
from modelmesh.auth.keys import KeyAuthenticator
from modelmesh.core.settings import Settings, get_settings
from modelmesh.persistence.db import make_engine, make_session_factory
from modelmesh.providers.catalog import load_catalog
from modelmesh.providers.registry import build_adapters


async def _bootstrap(name: str, env: str) -> None:
    """Create a tenant and a first key. The key is printed once, for the operator to store."""
    from redis.asyncio import Redis

    settings = get_settings()
    settings.require_runtime_secrets()
    engine = make_engine(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    sessions = make_session_factory(engine)
    admin = AdminService(
        sessions,
        KeyAuthenticator(sessions, redis, settings.api_key_pepper),
        settings.api_key_pepper,
        KeyDefaults(
            settings.default_key_rpm,
            settings.default_key_tpm,
            settings.default_key_concurrency,
            settings.default_key_daily_budget_usd,
        ),
    )
    try:
        tenant = await admin.create_tenant(name)
        key = await admin.create_key(tenant["id"], name=f"{name}-bootstrap", env=env)
        print(json.dumps({"tenant_id": tenant["id"], "key_id": key["id"], "api_key": key["key"]}))
    finally:
        await redis.aclose()
        await engine.dispose()


async def _catalog_verify(settings: Settings | None = None) -> int:
    """Compare config/models.yaml with each configured provider's /models. Non-zero on drift."""
    settings = settings or get_settings()
    catalog = load_catalog(settings.model_catalog_path)
    adapters = build_adapters(settings)
    problems = 0
    checked = 0
    for provider, adapter in adapters.items():
        try:
            listed = {model.id: model for model in await adapter.list_models()}
        except Exception as exc:
            print(f"{provider}: could not list models ({type(exc).__name__})")
            problems += 1
            continue
        for model in catalog.models.values():
            if model.provider != provider:
                continue
            checked += 1
            info = listed.get(model.provider_model)
            if info is None or not info.active:
                print(f"{model.id}: not offered by the provider")
                problems += 1
            elif info.context_window and model.context_window > info.context_window:
                print(f"{model.id}: catalog context window exceeds the provider's ({info.context_window})")
                problems += 1
        await adapter.aclose()
    print(f"checked {checked} models from {len(adapters)} configured providers; {problems} problem(s)")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="modelmesh")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap", help="create a tenant and its first API key")
    bootstrap.add_argument("--name", default="default")
    bootstrap.add_argument("--env", choices=["live", "test"], default="live")
    catalog = commands.add_parser("catalog", help="model catalog tools")
    catalog.add_argument("action", choices=["verify"])
    args = parser.parse_args()
    if sys.platform == "win32":
        # psycopg's async mode cannot use the default Windows event loop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if args.command == "bootstrap":
        asyncio.run(_bootstrap(args.name, args.env))
    else:
        raise SystemExit(asyncio.run(_catalog_verify()))
