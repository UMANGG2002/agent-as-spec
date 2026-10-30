#!/usr/bin/env python
"""Run agent evaluations from the command line / CI.

    # run a stored set (or a JSONL file kept in git) against the live model
    python scripts/eval.py run --tenant acme --set golden-quotes --gate
    python scripts/eval.py run --tenant acme --file evals/golden/quotes.jsonl --gate

    # deterministic run with the scripted model (no API key, no flakiness)
    python scripts/eval.py run --tenant acme --set synthetic --model scripted --no-judge --gate

    python scripts/eval.py generate --tenant acme --set synthetic     # cases from approved specs
    python scripts/eval.py planner  --tenant acme                      # evaluate the planner
    python scripts/eval.py baseline --tenant acme --run evr_...        # bless a run as the baseline
    python scripts/eval.py runs     --tenant acme [--set ID]
    python scripts/eval.py export   --tenant acme --set ID > cases.jsonl

`--gate` makes the exit code 1 when the regression gate fails (a previously
passing case now fails, the pass rate dropped, or the judge score fell past
tolerance), so it can block a merge. With no baseline yet, the gate is "every
hard check passes".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.evals import generator, golden, judges, planner_eval, service, store  # noqa: E402


def _model(name: str, transfer_to: str):
    if name == "scripted":
        from app.testing.scripted_llm import ScriptedToolCallingLlm

        return ScriptedToolCallingLlm(transfer_to=[x for x in transfer_to.split(",") if x])
    if not settings.planner_is_live:
        sys.exit("no live model configured (GOOGLE_API_KEY or LLM_BASE_URL + LLM_API_KEY); use --model scripted")
    return settings.live_model


async def _main(args: argparse.Namespace) -> int:
    from app.db import init_db

    await init_db()
    tenant = args.tenant
    ident = service.eval_identity(tenant)

    if args.cmd == "run":
        if args.file:
            text = Path(args.file).read_text()
            eval_set = golden.load_jsonl(text, tenant_id=tenant, set_id=args.set or Path(args.file).stem)
            await store.save_set(eval_set)
        else:
            eval_set = await store.get_set(tenant, args.set)
            if eval_set is None:
                sys.exit(f"eval set {args.set!r} not found for tenant {tenant!r}")
        report = await service.run_set(
            eval_set, ident, model=_model(args.model, args.transfer_to),
            judge=None if args.no_judge else judges.get_judge(), trigger=args.trigger,
            tags=args.tags.split(",") if args.tags else None, concurrency=args.concurrency)
        print(service.format_report(report))
        if args.json:
            Path(args.json).write_text(report.model_dump_json(indent=2))
        if args.make_baseline:
            await store.set_baseline(tenant, report.run_id)
            print(f"baseline for {eval_set.set_id}: {report.run_id}")
        return 1 if args.gate and report.regression and not report.regression.ok else 0

    if args.cmd == "generate":
        from app.specs import store as specs_store

        specs = await specs_store.list_specs(tenant)
        if not specs:
            sys.exit("no approved specs for this tenant")
        s = await store.save_set(generator.generate_set(tenant, specs, set_id=args.set))
        print(f"set {s.set_id} v{s.version}: {len(s.cases)} cases from {len(specs)} spec(s)")
        return 0

    if args.cmd == "planner":
        from app.registry import store as registry_store

        cases = planner_eval.generate_planner_cases(await registry_store.visible_capabilities(tenant),
                                                    tenant_id=tenant, limit=args.limit)
        report = await planner_eval.run_planner_cases(ident, cases, trigger=args.trigger)
        await store.save_run(report)
        print(service.format_report(report))
        return 1 if args.gate and report.summary.pass_rate < args.min_pass_rate else 0

    if args.cmd == "baseline":
        ok = await store.set_baseline(tenant, args.run)
        print("baseline set" if ok else "run not found")
        return 0 if ok else 1

    if args.cmd == "runs":
        for r in await store.list_runs(tenant, args.set):
            print(f"{r['run_id']}  {r['set_id']:<20} {r['trigger']:<14} pass {r['pass_rate']:.0%}  "
                  f"score {r['mean_score']:.3f}  {'BASELINE' if r['is_baseline'] else ''}  {r['created_at']}")
        return 0

    if args.cmd == "export":
        s = await store.get_set(tenant, args.set)
        if s is None:
            sys.exit("eval set not found")
        sys.stdout.write(golden.dump_jsonl(s))
        return 0
    return 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "generate", "planner", "baseline", "runs", "export"):
        sp = sub.add_parser(name)
        sp.add_argument("--tenant", required=True)
        sp.add_argument("--set", default=None if name in ("run", "runs", "export") else "synthetic")
        if name == "run":
            sp.add_argument("--file", help="JSONL cases file (kept in git); imported into --set (default: file name)")
            sp.add_argument("--model", choices=["live", "scripted"], default="live")
            sp.add_argument("--transfer-to", default="", help="scripted model: router sub-agent names to hand off to")
            sp.add_argument("--no-judge", action="store_true")
            sp.add_argument("--tags")
            sp.add_argument("--concurrency", type=int)
            sp.add_argument("--json", help="write the full report to this file")
            sp.add_argument("--make-baseline", action="store_true")
        if name in ("run", "planner"):
            sp.add_argument("--gate", action="store_true", help="exit 1 when the gate fails")
            sp.add_argument("--trigger", default="ci")
        if name == "planner":
            sp.add_argument("--limit", type=int, default=30)
            sp.add_argument("--min-pass-rate", type=float, default=0.9)
        if name == "baseline":
            sp.add_argument("--run", required=True)
    args = p.parse_args()
    if args.cmd == "run" and not (args.set or args.file):
        p.error("run needs --set or --file")
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
