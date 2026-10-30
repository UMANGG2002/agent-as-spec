"""Evaluation API (tenant-scoped).

    GET    /v1/evals/sets                      list sets
    PUT    /v1/evals/sets/{id}                 create/replace a set (cases inline)
    GET    /v1/evals/sets/{id}                 one set
    DELETE /v1/evals/sets/{id}
    POST   /v1/evals/sets/{id}/import          JSONL body -> cases (upsert)
    GET    /v1/evals/sets/{id}/export          JSONL
    POST   /v1/evals/sets/generate             synthetic cases from approved specs
    POST   /v1/evals/sets/{id}/promote         a real session -> draft golden case
    POST   /v1/evals/sets/{id}/runs            run it now (<= 50 cases; use the CLI for more)
    GET    /v1/evals/runs                      history; GET /v1/evals/runs/{run}; POST .../baseline
    POST   /v1/evals/planner/runs              evaluate the planner (generated or given cases)
    GET    /v1/evals/specs/{spec_id}/status    latest auto-eval gate for a spec
    GET    /v1/evals/online                    sampled live-traffic scores + worst conversations

Running needs the `eval:run` scope (or `*`) and a model: the live one, or
`"model": "scripted"` outside staging/production. The agents run in the
CALLER's tenant with the caller's scopes (as the synthetic user `eval-bot`) -- an eval never
gets more access than its caller.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.config import settings
from app.evals import auto, generator, golden, judges, online, planner_eval, service, store
from app.evals.schemas import EvalCase, EvalSet

router = APIRouter(prefix="/v1/evals", tags=["evals"])
MAX_API_CASES = 50


def _can_run(identity: IdentityContext) -> None:
    if not identity.has_scope("eval:run"):
        raise HTTPException(403, "running evals needs the 'eval:run' scope")


def _eval_identity(identity: IdentityContext) -> IdentityContext:
    return service.eval_identity(identity.tenant_id, scopes=identity.scopes)


def _model(name: str):
    if name == "scripted":
        if settings.is_production_like:
            raise HTTPException(400, "the scripted model is not available in staging/production")
        from app.testing.scripted_llm import ScriptedToolCallingLlm

        return ScriptedToolCallingLlm()
    if not settings.planner_is_live:
        raise HTTPException(409, "no live model is configured (set GOOGLE_API_KEY or LLM_BASE_URL/LLM_API_KEY), "
                                 "or pass model='scripted' in a non-production environment")
    return settings.live_model


class SetBody(BaseModel):
    name: str
    description: str = ""
    kind: str = "golden"
    cases: list[EvalCase] = Field(default_factory=list)


class GenerateBody(BaseModel):
    set_id: str = "synthetic"
    spec_ids: list[str] | None = None
    include_router: bool = True


class PromoteBody(BaseModel):
    session_id: str
    case_id: str | None = None


class RunBody(BaseModel):
    model: str = "live"  # live | scripted
    judge: bool = True
    tags: list[str] | None = None
    trigger: str = "manual"
    compare_to_baseline: bool = True


class PlannerRunBody(BaseModel):
    cases: list[planner_eval.PlannerCase] | None = None  # None = generate one retrieval case per capability
    limit: int = 20


@router.get("/sets")
async def list_sets(identity: IdentityContext = Depends(get_identity)):
    return await store.list_sets(identity.tenant_id)


@router.put("/sets/{set_id}")
async def put_set(set_id: str, body: SetBody, identity: IdentityContext = Depends(get_identity)):
    ids = [c.case_id for c in body.cases]
    if len(ids) != len(set(ids)):
        raise HTTPException(422, "duplicate case_id")
    s = EvalSet(set_id=set_id, tenant_id=identity.tenant_id, name=body.name, description=body.description,
                kind=body.kind, cases=body.cases)  # type: ignore[arg-type]
    return await store.save_set(s)


@router.get("/sets/{set_id}")
async def get_set(set_id: str, identity: IdentityContext = Depends(get_identity)):
    s = await store.get_set(identity.tenant_id, set_id)
    if s is None:
        raise HTTPException(404, "eval set not found")
    return s


@router.delete("/sets/{set_id}", status_code=204)
async def delete_set(set_id: str, identity: IdentityContext = Depends(get_identity)):
    if not await store.delete_set(identity.tenant_id, set_id):
        raise HTTPException(404, "eval set not found")


@router.post("/sets/{set_id}/import")
async def import_jsonl(set_id: str, request: Request, identity: IdentityContext = Depends(get_identity)):
    try:
        parsed = golden.load_jsonl((await request.body()).decode(), tenant_id=identity.tenant_id, set_id=set_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    s = await golden.add_cases(identity.tenant_id, set_id, parsed.cases)
    return {"set_id": s.set_id, "version": s.version, "n_cases": len(s.cases)}


@router.get("/sets/{set_id}/export", response_class=PlainTextResponse)
async def export_jsonl(set_id: str, identity: IdentityContext = Depends(get_identity)):
    s = await store.get_set(identity.tenant_id, set_id)
    if s is None:
        raise HTTPException(404, "eval set not found")
    return golden.dump_jsonl(s)


@router.post("/sets/generate")
async def generate(body: GenerateBody, identity: IdentityContext = Depends(get_identity)):
    from app.specs import store as specs_store

    specs = await specs_store.list_specs(identity.tenant_id)
    if body.spec_ids:
        specs = [s for s in specs if s.spec_id in set(body.spec_ids)]
    if not specs:
        raise HTTPException(404, "no approved specs to generate cases from")
    s = await store.save_set(generator.generate_set(identity.tenant_id, specs, set_id=body.set_id,
                                                    include_router=body.include_router))
    return {"set_id": s.set_id, "version": s.version, "n_cases": len(s.cases)}


@router.post("/sets/{set_id}/promote")
async def promote(set_id: str, body: PromoteBody, identity: IdentityContext = Depends(get_identity)):
    try:
        case = await golden.promote_session(identity.tenant_id, body.session_id, case_id=body.case_id, set_id=set_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return case


@router.post("/sets/{set_id}/runs")
async def run(set_id: str, body: RunBody, identity: IdentityContext = Depends(get_identity)):
    _can_run(identity)
    s = await store.get_set(identity.tenant_id, set_id)
    if s is None:
        raise HTTPException(404, "eval set not found")
    if len(s.cases) > MAX_API_CASES:
        raise HTTPException(413, f"{len(s.cases)} cases is more than the API runs inline ({MAX_API_CASES}); "
                                 "use scripts/eval.py or pass tags to run a subset")
    model = _model(body.model)
    report = await service.run_set(s, _eval_identity(identity), model=model,
                                   judge=judges.get_judge() if body.judge else None, trigger=body.trigger,
                                   compare_to_baseline=body.compare_to_baseline, tags=body.tags)
    return report


@router.get("/runs")
async def list_runs(set_id: str | None = None, limit: int = 30, identity: IdentityContext = Depends(get_identity)):
    return await store.list_runs(identity.tenant_id, set_id, limit)


@router.get("/runs/{run_id}")
async def get_run(run_id: str, identity: IdentityContext = Depends(get_identity)):
    r = await store.get_run(identity.tenant_id, run_id)
    if r is None:
        raise HTTPException(404, "eval run not found")
    return r


@router.post("/runs/{run_id}/baseline")
async def make_baseline(run_id: str, identity: IdentityContext = Depends(get_identity)):
    _can_run(identity)
    if not await store.set_baseline(identity.tenant_id, run_id):
        raise HTTPException(404, "eval run not found")
    return {"run_id": run_id, "is_baseline": True}


@router.post("/planner/runs")
async def run_planner(body: PlannerRunBody = Body(default_factory=PlannerRunBody),
                      identity: IdentityContext = Depends(get_identity)):
    _can_run(identity)
    cases = body.cases
    if cases is None:
        from app.registry import store as registry_store

        cases = planner_eval.generate_planner_cases(await registry_store.visible_capabilities(identity.tenant_id),
                                                    tenant_id=identity.tenant_id, limit=body.limit)
    if not cases:
        raise HTTPException(404, "no capabilities to build planner cases from")
    return await planner_eval.run_planner_cases(_eval_identity(identity), cases[:MAX_API_CASES])


@router.get("/specs/{spec_id}/status")
async def spec_status(spec_id: str, identity: IdentityContext = Depends(get_identity)):
    return await auto.spec_status(identity.tenant_id, spec_id)


@router.get("/online")
async def online_summary(spec_id: str | None = None, identity: IdentityContext = Depends(get_identity)):
    return await online.summary(identity.tenant_id, spec_id)
