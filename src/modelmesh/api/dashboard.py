"""Operator dashboard (M7): server-rendered pages over the same service-layer reads as the public API.

Plain HTML forms and Jinja2 templates; no JavaScript and no build step. Pages read Postgres, Redis and in-process
state only, and never call a provider.

Access needs the admin token, sent either as a bearer header or once through the login form. The form sets a session
cookie holding an HMAC derived from the token, never the token itself. State-changing forms also carry a CSRF value
derived the same way, and the cookie is SameSite=Strict.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import parse_qs, urlencode

import jinja2
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from modelmesh.accounting.queries import GROUP_COLUMNS, AccountingQueries, UsageQuery
from modelmesh.api.routes import MAX_USAGE_RANGE, POLICIES
from modelmesh.auth.admin import AdminService, KeyDefaults
from modelmesh.contract.schemas import RoutingOptions
from modelmesh.core.errors import ApiError
from modelmesh.pipeline.services import Services
from modelmesh.pipeline.status import acceptance_criteria, model_entries, provider_entries
from modelmesh.routing.solver import DEFAULT_MIN_QUALITY, KNOWN_TASK_TYPES, TASK_TIER

PREFIX = "/dashboard"
SESSION_COOKIE = "modelmesh_dashboard"
DEFAULT_WINDOW = timedelta(days=7)
PAGE_SIZE = 50
TEMPLATES = Path(__file__).parent / "templates"

_TENANT_ID = re.compile(r"^ten_[A-Za-z0-9]{8,40}$")
_KEY_ID = re.compile(r"^key_[A-Za-z0-9]{8,40}$")
_REQUEST_ID = re.compile(r"^mmr_[A-Za-z0-9]{8,40}$")
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9_:.-]{1,128}$")
_CURSOR = re.compile(r"^[A-Za-z0-9_=-]{1,256}$")

# Columns of the usage tables: (field in the /v1/usage row, heading, display kind).
USAGE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("requests", "Requests", "int"),
    ("succeeded", "Succeeded", "int"),
    ("failed", "Failed", "int"),
    ("cache_hits", "Cache hits", "int"),
    ("fallback_events", "Fallbacks", "int"),
    ("prompt_tokens_provider_reported", "Prompt tokens, reported", "int"),
    ("completion_tokens_provider_reported", "Completion tokens, reported", "int"),
    ("prompt_tokens_estimated", "Prompt tokens, estimated", "int"),
    ("completion_tokens_estimated", "Completion tokens, estimated", "int"),
    ("estimated_cost_usd", "Cost, estimated (USD)", "usd"),
    ("provider_reported_cost_usd", "Cost, provider-reported (USD)", "usd"),
    ("provider_reported_cost_coverage", "Reported-cost coverage", "ratio"),
    ("latency_p50_ms", "p50 latency (ms)", "int"),
    ("latency_p95_ms", "p95 latency (ms)", "int"),
)
PROVIDER_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("status", "Status", "text"),
    ("circuit_state", "Breaker", "text"),
    ("error_rate_5m", "Error rate, 5 min", "ratio"),
    ("p50_latency_ms", "p50 latency (ms)", "int"),
    ("p95_latency_ms", "p95 latency (ms)", "int"),
    ("models_available", "Models available", "int"),
)
RELIABILITY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("requests", "Requests", "int"),
    ("succeeded", "Succeeded", "int"),
    ("failed", "Failed", "int"),
    ("fallback_events", "Fallbacks", "int"),
    ("latency_p50_ms", "p50 latency (ms)", "int"),
    ("latency_p95_ms", "p95 latency (ms)", "int"),
)


class DashboardLogin(Exception):
    """Raised when a page needs the admin token; rendered as the login page with 401."""


def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["show"] = show
    env.filters["raw"] = raw
    return env


# ------------------------------------------------------------------ formatting
def raw(value: Any) -> str:
    """The exact value behind a displayed number, kept in `data-value` so that tests can compare it with the API."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def show(value: Any, kind: str = "text") -> str:
    if value is None:
        return "-"
    if kind == "usd":
        return f"{float(value):.6f}"
    if kind == "ratio":
        return f"{float(value) * 100:.1f}%"
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(value)


templates = Jinja2Templates(env=_environment())
dashboard_router = APIRouter(prefix=PREFIX, include_in_schema=False)


# ------------------------------------------------------------------ access
def _services(request: Request) -> Services:
    result: Services = request.app.state.services
    return result


def _derive(token: str, purpose: str) -> str:
    return hmac.new(token.encode(), f"modelmesh-dashboard:{purpose}".encode(), hashlib.sha256).hexdigest()


def _admin_token(svc: Services) -> str:
    return svc.settings.admin_token.get_secret_value()


def _is_authorised(request: Request, svc: Services) -> bool:
    token = _admin_token(svc)
    if not token:
        return False
    header = request.headers.get("authorization") or ""
    if hmac.compare_digest(header.encode(), f"Bearer {token}".encode()):
        return True
    cookie = request.cookies.get(SESSION_COOKIE) or ""
    return hmac.compare_digest(cookie.encode(), _derive(token, "session").encode())


def require_dashboard_admin(request: Request) -> Services:
    svc = _services(request)
    if not _is_authorised(request, svc):
        raise DashboardLogin
    return svc


def _admin_service(svc: Services) -> AdminService:
    settings = svc.settings
    return AdminService(
        svc.sessions,
        svc.authenticator,
        settings.api_key_pepper,
        KeyDefaults(
            settings.default_key_rpm,
            settings.default_key_tpm,
            settings.default_key_concurrency,
            settings.default_key_daily_budget_usd,
        ),
    )


def _secure_headers(response: Response) -> Response:
    response.headers["cache-control"] = "no-store"
    response.headers["content-security-policy"] = (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    )
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["referrer-policy"] = "no-referrer"
    return response


def render(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> Response:
    svc = _services(request)
    token = _admin_token(svc)
    base = {"prefix": PREFIX, "csrf": _derive(token, "csrf") if token else "", "signed_in": True}
    response = templates.TemplateResponse(request, name, base | context, status_code=status_code)
    return _secure_headers(response)


def _login_page(request: Request, *, status_code: int, message: str | None = None) -> Response:
    configured = bool(_admin_token(_services(request)))
    context = {"message": message, "configured": configured, "signed_in": False, "nav": None}
    return render(request, "login.html", context, status_code=status_code if configured else 503)


async def _form(request: Request) -> dict[str, str]:
    body = (await request.body()).decode("utf-8", errors="replace")
    return {name: values[0] for name, values in parse_qs(body, keep_blank_values=True).items()}


def _check_csrf(request: Request, form: dict[str, str]) -> None:
    token = _admin_token(_services(request))
    if not hmac.compare_digest(form.get("csrf", "").encode(), _derive(token, "csrf").encode()):
        raise ApiError("permission_denied", "The form token is missing or invalid.", code="invalid_value")


def install_dashboard(app: FastAPI) -> None:
    async def on_login(request: Request, exc: Exception) -> Response:
        return _login_page(request, status_code=401)

    app.add_exception_handler(DashboardLogin, on_login)
    app.include_router(dashboard_router)


AdminServices = Annotated[Services, Depends(require_dashboard_admin)]


# ------------------------------------------------------------------ helpers
class BadInput(ValueError):
    pass


def _parse_time(name: str, value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BadInput(f"Invalid value for '{name}'.") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _window(request: Request) -> tuple[datetime, datetime]:
    end = _parse_time("to", request.query_params.get("to")) or datetime.now(UTC).replace(microsecond=0)
    start = _parse_time("from", request.query_params.get("from")) or end - DEFAULT_WINDOW
    if end <= start or end - start > MAX_USAGE_RANGE:
        raise BadInput("The range must be positive and at most 92 days.")
    return start, end


async def _tenant_scope(request: Request, svc: Services) -> tuple[list[dict[str, Any]], str | None]:
    tenants = await _admin_service(svc).list_tenants()
    wanted = request.query_params.get("tenant_id")
    if wanted is not None and not _TENANT_ID.match(wanted):
        raise BadInput("Invalid tenant id.")
    known = {tenant["id"] for tenant in tenants}
    if wanted is not None and wanted not in known:
        raise BadInput("Unknown tenant.")
    return tenants, wanted or (tenants[0]["id"] if tenants else None)


def _bad_input(request: Request, nav: str, error: BadInput) -> Response:
    return render(request, "error.html", {"nav": nav, "message": str(error)}, status_code=400)


def trace_url(svc: Services, trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    return f"{svc.settings.jaeger_ui_url.rstrip('/')}/trace/{trace_id}"


# ------------------------------------------------------------------ sign-in
@dashboard_router.get("")
async def dashboard_root() -> Response:
    return RedirectResponse(f"{PREFIX}/usage", status_code=303)


@dashboard_router.get("/login")
async def login_form(request: Request) -> Response:
    return _login_page(request, status_code=200)


@dashboard_router.post("/login")
async def login(request: Request) -> Response:
    svc = _services(request)
    token = _admin_token(svc)
    supplied = (await _form(request)).get("token", "")
    if not token or not hmac.compare_digest(supplied.encode(), token.encode()):
        return _login_page(request, status_code=401, message="That token is not valid.")
    response = RedirectResponse(f"{PREFIX}/usage", status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        _derive(token, "session"),
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path=PREFIX,
    )
    return _secure_headers(response)


@dashboard_router.post("/logout")
async def logout(request: Request, svc: AdminServices) -> Response:
    _check_csrf(request, await _form(request))
    response = RedirectResponse(f"{PREFIX}/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path=PREFIX)
    return _secure_headers(response)


# ------------------------------------------------------------------ usage and cost
@dashboard_router.get("/usage")
async def usage_page(request: Request, svc: AdminServices) -> Response:
    try:
        tenants, tenant_id = await _tenant_scope(request, svc)
        start, end = _window(request)
        group_by = request.query_params.get("group_by", "model")
        if group_by not in GROUP_COLUMNS:
            raise BadInput("Invalid value for 'group_by'.")
    except BadInput as error:
        return _bad_input(request, "usage", error)
    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] | None = None
    if tenant_id is not None:
        rows, totals = await AccountingQueries(svc.sessions).usage(UsageQuery(tenant_id, start, end, group_by))
    context = {
        "nav": "usage",
        "tenants": tenants,
        "tenant_id": tenant_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "group_by": group_by,
        "group_options": list(GROUP_COLUMNS),
        "fields": USAGE_FIELDS,
        "rows": rows,
        "totals": totals,
    }
    return render(request, "usage.html", context)


# ------------------------------------------------------------------ latency and reliability
@dashboard_router.get("/reliability")
async def reliability_page(request: Request, svc: AdminServices) -> Response:
    try:
        tenants, tenant_id = await _tenant_scope(request, svc)
        start, end = _window(request)
    except BadInput as error:
        return _bad_input(request, "reliability", error)
    rows: list[dict[str, Any]] = []
    if tenant_id is not None:
        rows, _ = await AccountingQueries(svc.sessions).usage(UsageQuery(tenant_id, start, end, "model"))
    models = [
        {
            "id": entry["id"],
            "provider": entry["modelmesh"]["provider"],
            "status": entry["modelmesh"]["status"],
            "circuit_state": svc.breakers.for_model(entry["id"]).state,
        }
        for entry in await model_entries(svc, None)
        if entry["modelmesh"]["kind"] == "model"
    ]
    context = {
        "nav": "reliability",
        "tenants": tenants,
        "tenant_id": tenant_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "providers": provider_entries(svc),
        "provider_fields": PROVIDER_FIELDS,
        "models": models,
        "rows": rows,
        "fields": RELIABILITY_FIELDS,
    }
    return render(request, "reliability.html", context)


# ------------------------------------------------------------------ request explorer
@dashboard_router.get("/requests")
async def requests_page(request: Request, svc: AdminServices) -> Response:
    params = request.query_params
    correlation_id = params.get("correlation_id") or None
    cursor = params.get("cursor") or None
    try:
        tenants, tenant_id = await _tenant_scope(request, svc)
        if correlation_id is not None and not _CORRELATION_ID.match(correlation_id):
            raise BadInput("Invalid value for 'correlation_id'.")
        if cursor is not None and not _CURSOR.match(cursor):
            raise BadInput("Invalid value for 'cursor'.")
    except BadInput as error:
        return _bad_input(request, "requests", error)
    records: list[dict[str, Any]] = []
    next_cursor: str | None = None
    traces: dict[str, str] = {}
    if tenant_id is not None:
        queries = AccountingQueries(svc.sessions)
        records, next_cursor = await queries.list_requests(
            tenant_id, correlation_id=correlation_id, start=None, end=None, limit=PAGE_SIZE, cursor=cursor
        )
        traces = await queries.trace_ids(tenant_id, [record["id"] for record in records])
    next_link = None
    if next_cursor is not None and tenant_id is not None:
        query = {"tenant_id": tenant_id, "cursor": next_cursor} | (
            {"correlation_id": correlation_id} if correlation_id else {}
        )
        next_link = f"{PREFIX}/requests?{urlencode(query)}"
    context = {
        "nav": "requests",
        "tenants": tenants,
        "tenant_id": tenant_id,
        "correlation_id": correlation_id or "",
        "records": [_summary(record, trace_url(svc, traces.get(record["id"]))) for record in records],
        "next_link": next_link,
    }
    return render(request, "requests.html", context)


def _summary(record: dict[str, Any], trace: str | None) -> dict[str, Any]:
    """The fields shown in the request list. Never includes retained content."""
    execution = record["modelmesh"] or {}
    metadata = record["metadata"] or {}
    return {
        "id": record["id"],
        "created_at": record["created_at"],
        "status": record["status"],
        "http_status": record["http_status"],
        "model": execution.get("model"),
        "latency_ms": (execution.get("latency_ms") or {}).get("total"),
        "estimated_usd": (execution.get("cost") or {}).get("estimated_usd"),
        "application": metadata.get("application"),
        "task_type": metadata.get("task_type"),
        "correlation_id": metadata.get("correlation_id"),
        "trace_url": trace,
    }


@dashboard_router.get("/requests/{request_id}")
async def request_detail(request_id: str, request: Request, svc: AdminServices) -> Response:
    try:
        tenants, tenant_id = await _tenant_scope(request, svc)
        if not _REQUEST_ID.match(request_id):
            raise BadInput("Invalid request id.")
    except BadInput as error:
        return _bad_input(request, "requests", error)
    queries = AccountingQueries(svc.sessions)
    record = await queries.get_request(tenant_id, request_id) if tenant_id is not None else None
    if record is None or tenant_id is None:
        return render(request, "error.html", {"nav": "requests", "message": "Unknown request id."}, status_code=404)
    traces = await queries.trace_ids(tenant_id, [request_id])
    execution = record["modelmesh"] or {}
    context = {
        "nav": "requests",
        "tenants": tenants,
        "tenant_id": tenant_id,
        "record": record,
        "summary": _summary(record, trace_url(svc, traces.get(request_id))),
        # A failed request has no execution metadata; its attempts travel in the error details instead.
        "attempts": execution.get("attempts") or ((record["error"] or {}).get("details") or {}).get("attempts") or [],
        "usage": execution.get("usage"),
        "cost": execution.get("cost"),
        "latency": execution.get("latency_ms"),
        # Content exists only when the key had content retention on and it has not expired (ADR-0010).
        "content": record["content"],
    }
    return render(request, "request_detail.html", context)


# ------------------------------------------------------------------ key management
async def _keys_context(request: Request, svc: Services, tenant_id: str | None) -> dict[str, Any]:
    tenants = await _admin_service(svc).list_tenants()
    keys = await _admin_service(svc).list_keys(tenant_id) if tenant_id is not None else []
    return {"nav": "keys", "tenants": tenants, "tenant_id": tenant_id, "keys": keys, "issued": None}


@dashboard_router.get("/keys")
async def keys_page(request: Request, svc: AdminServices) -> Response:
    try:
        _, tenant_id = await _tenant_scope(request, svc)
    except BadInput as error:
        return _bad_input(request, "keys", error)
    return render(request, "keys.html", await _keys_context(request, svc, tenant_id))


async def _form_tenant(request: Request, svc: Services, form: dict[str, str]) -> str:
    tenant_id = form.get("tenant_id", "")
    known = {tenant["id"] for tenant in await _admin_service(svc).list_tenants()}
    if not _TENANT_ID.match(tenant_id) or tenant_id not in known:
        raise BadInput("Unknown tenant.")
    return tenant_id


@dashboard_router.post("/keys")
async def create_key(request: Request, svc: AdminServices) -> Response:
    form = await _form(request)
    _check_csrf(request, form)
    try:
        tenant_id = await _form_tenant(request, svc, form)
        name = form.get("name", "").strip()
        env = form.get("env", "live")
        if not 1 <= len(name) <= 200:
            raise BadInput("A key name of 1 to 200 characters is required.")
        if env not in ("live", "test"):
            raise BadInput("Invalid value for 'env'.")
    except BadInput as error:
        return _bad_input(request, "keys", error)
    created = await _admin_service(svc).create_key(tenant_id, name=name, env=env)
    context = await _keys_context(request, svc, tenant_id)
    context["issued"] = {"id": created["id"], "key": created["key"], "action": "created"}
    return render(request, "keys.html", context)


@dashboard_router.post("/keys/{key_id}/{action}")
async def change_key(key_id: str, action: str, request: Request, svc: AdminServices) -> Response:
    form = await _form(request)
    _check_csrf(request, form)
    try:
        tenant_id = await _form_tenant(request, svc, form)
        if not _KEY_ID.match(key_id) or action not in ("revoke", "rotate"):
            raise BadInput("Unknown key action.")
        if key_id not in {key["id"] for key in await _admin_service(svc).list_keys(tenant_id)}:
            raise BadInput("Unknown key.")
    except BadInput as error:
        return _bad_input(request, "keys", error)
    admin = _admin_service(svc)
    if action == "revoke":
        await admin.revoke_key(key_id)
        return _secure_headers(RedirectResponse(f"{PREFIX}/keys?{urlencode({'tenant_id': tenant_id})}", 303))
    rotated = await admin.rotate_key(key_id)
    context = await _keys_context(request, svc, tenant_id)
    context["issued"] = {"id": key_id, "key": rotated["key"], "action": "rotated"}
    return render(request, "keys.html", context)


# ------------------------------------------------------------------ routing policy
@dashboard_router.get("/routing")
async def routing_page(request: Request, svc: AdminServices) -> Response:
    quality = svc.estimator.quality
    catalog = svc.registry.catalog.models
    tasks = [
        {
            "task_type": task,
            "rules_tier": TASK_TIER[task],
            "calibrated": quality.is_task_calibrated(task),
            "acceptance_criterion": acceptance_criteria(task),
        }
        for task in KNOWN_TASK_TYPES
    ]
    models = [
        {
            "id": entry["id"],
            "provider": entry["modelmesh"]["provider"],
            "family": entry["modelmesh"]["family"],
            "size_tier": catalog[entry["id"]].size_tier,
            "data_use": entry["modelmesh"]["data_use"],
            "capabilities": entry["modelmesh"]["capabilities"],
            "pricing": entry["modelmesh"]["pricing"],
            "status": entry["modelmesh"]["status"],
        }
        for entry in await model_entries(svc, None)
        if entry["modelmesh"]["kind"] == "model" and entry["id"] in catalog
    ]
    context = {
        "nav": "routing",
        "router_version": quality.router_version,
        "default_policy": RoutingOptions.model_fields["policy"].default,
        "policies": POLICIES,
        "default_min_quality": DEFAULT_MIN_QUALITY,
        "tasks": tasks,
        "models": models,
        "pricing_version": svc.pricing.version,
    }
    return render(request, "routing.html", context)


# ------------------------------------------------------------------ benchmark comparison
@dashboard_router.get("/benchmarks")
async def benchmarks_page(request: Request, svc: AdminServices) -> Response:
    return render(request, "benchmarks.html", {"nav": "benchmarks"})
