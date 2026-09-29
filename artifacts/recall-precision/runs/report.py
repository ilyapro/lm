#!/usr/bin/env python3
"""Build results.json and report.md from the frozen holdout runs (prereg.md).

Inputs (all under artifacts/recall-precision/runs/): holdout-{host}-{a,b}.json
(driver output, one baseline per file; arms are compared with the baseline of
their own file, which is the same replay on the same events), census-{host}-
holdout.json and census-{host}-trainval.json. Hosts are never merged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent
HOSTS = ("sfx", "alt")
KEEP = (
    "events", "full_slots_per_event", "stub_slots_per_event", "irrelevant", "used", "hub_top3_share",
    "hub_top3_slots", "dup_schema_slots", "chars_per_event", "rank1_removed", "chars_saved_per_event",
    "chars_saved_share", "vs_baseline", "vs_baseline_by_rank", "by_recorded_rank", "new_entrant_quality",
    "delivered_set_quality", "future_candidates_hidden", "labelled_created_after_cutoff_excluded",
)


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text())


def pct(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def host_block(host: str) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    for part in ("a", "b"):
        run = load(f"holdout-{host}-{part}.json")
        meta = {k: run[k] for k in ("segment", "cutoff", "events", "snapshot_sha256", "hubs_train_defined")}
        for name, summary in run["summary"].items():
            key = f"baseline_{part}" if name == "baseline" else name
            arms[key] = {"env": run["arms"][name], **{k: summary.get(k) for k in KEEP if k in summary}}
    census = {
        window: {
            name: setting["all"] | {"delivered": setting["delivered"]}
            for name, setting in load(f"census-{host}-{window}.json")["settings"].items()
        }
        | {"_events": load(f"census-{host}-{window}.json")["events"], "_applicable": load(f"census-{host}-{window}.json")["applicable"]}
        for window in ("trainval", "holdout")
    }
    return {**meta, "arms": arms, "demotion_census": census}


def verdicts(host: str, block: dict[str, Any]) -> list[dict[str, Any]]:
    arms = block["arms"]
    base_q = arms["baseline_a"]["delivered_set_quality"]["excess"]
    base_q_b = arms["baseline_b"]["delivered_set_quality"]["excess"]
    gate = arms["gate"]
    hub = arms["hub"]
    out: list[dict[str, Any]] = []

    def add(cid: str, text: str, value: Any, passed: bool | None) -> None:
        out.append({"id": cid, "condition": text, "value": value, "verdict": "PASS" if passed else ("REPORT" if passed is None else "FAIL")})

    cut = gate["vs_baseline"]["irrelevant_full_cut"]
    add("G1", "gate irrelevant_full_cut >= 25%", cut, cut is not None and cut >= 0.25)
    lost = gate["vs_baseline"]["used_full_lost"]
    add("G2", "gate used_full_lost <= 12%", lost, lost is not None and lost <= 0.12)
    add("G3", "gate rank1_removed == 0", gate["rank1_removed"], gate["rank1_removed"] == 0)
    hub_cut = hub["vs_baseline"]["hub_top3_cut"]
    hub_lost = hub["vs_baseline"]["used_full_lost"]
    if host == "sfx":
        add("H1", "hub hub_top3_cut >= 80%", hub_cut, hub_cut is not None and hub_cut >= 0.80)
        add("H2", "hub used_full_lost <= 3%", hub_lost, hub_lost is not None and hub_lost <= 0.03)
        share = block["demotion_census"]["holdout"]["fc080_mw1"]["share_gt_0_9"]
        add("D1", "fc080_mw1 census share m>0.9 < 30% (holdout window)", share, share < 0.30)
    else:
        add("H3", "hub cut / used loss (reported)", {"hub_top3_cut": hub_cut, "used_full_lost": hub_lost}, None)
        share = block["demotion_census"]["holdout"]["fc080_mw1"]["share_gt_0_9"]
        add("D2", "fc080_mw1 census share m>0.9 (reported)", share, None)
    dup = arms["dedup"]["dup_schema_slots"]
    add("S1", "dedup dup_schema_slots == 0", dup, dup == 0)
    for arm, base in (("gate", base_q), ("hub", base_q_b), ("combined", base_q)):
        quality = arms[arm]["new_entrant_quality"]
        value = {"new_entrant_excess": quality["excess"], "graded": quality["graded"], "baseline_delivered_excess": base}
        if quality["graded"] < 10:
            out.append({"id": f"Q1-{arm}", "condition": f"{arm} new-entrant excess >= baseline delivered-set excess", "value": value, "verdict": "INSUFFICIENT_N"})
        else:
            add(f"Q1-{arm}", f"{arm} new-entrant excess >= baseline delivered-set excess", value, quality["excess"] >= base)
    return out


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return pct(value)
    if isinstance(value, dict):
        return ", ".join(f"{k} {fmt_value(v)}" for k, v in value.items())
    return str(value)


def render(results: dict[str, Any]) -> str:
    lines = ["# Recall precision valves: holdout report", ""]
    lines += [
        "Pre-registered in `prereg.md` (commits 86f0e7b and a1d0255, before this run). One frozen run per host",
        "on the holdout segment, all events, production env. Numbers are per host and never merged. Every",
        "effect is against the replayed baseline arm on the same events.",
        "",
    ]
    for host in HOSTS:
        block = results["hosts"][host]
        arms = block["arms"]
        lines += [f"## {host}", "", f"- holdout events {block['events']}, cutoff {block['cutoff']}, train-defined hubs {block['hubs_train_defined']}", ""]
        lines += ["### Verdicts", "", "| id | condition | value | verdict |", "|---|---|---|---|"]
        for item in block["verdicts"]:
            lines.append(f"| {item['id']} | {item['condition']} | {fmt_value(item['value'])} | **{item['verdict']}** |")
        lines += ["", "### Arms", ""]
        lines += [
            "| arm | full/ev | irr full | irr cut | used full | used lost | rank1 removed | hub top3 | hub cut | dup schema | chars/ev | chars saved | new entrants graded | entrant excess |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, arm in arms.items():
            vs = arm.get("vs_baseline") or {}
            quality = arm.get("new_entrant_quality") or arm.get("delivered_set_quality") or {}
            lines.append(
                f"| {name} | {arm['full_slots_per_event']} | {arm['irrelevant']['delivered_full']} | {pct(vs.get('irrelevant_full_cut'))} "
                f"| {arm['used']['delivered_full']} | {pct(vs.get('used_full_lost'))} | {arm.get('rank1_removed', '—')} "
                f"| {arm['hub_top3_slots']} | {pct(vs.get('hub_top3_cut'))} | {arm['dup_schema_slots']} | {arm['chars_per_event']} "
                f"| {pct(arm.get('chars_saved_share'))} | {quality.get('graded')} | {quality.get('excess')} |"
            )
        lines += [
            "",
            "`baseline_a` is the baseline for gate, gate_alt_T and combined, and `baseline_b` for hub, demote and dedup.",
            "The two are the same replay in separate processes. For a baseline row, the entrant columns show the quality of the whole delivered set.",
            "",
            "### By recorded rank (irr cut / used lost vs baseline; n = baseline full slots)",
            "",
            "| arm | 1 | 2 | 3 | 4-5 | 6-10 |",
            "|---|---|---|---|---|---|",
        ]
        for name, arm in arms.items():
            ranks = arm.get("vs_baseline_by_rank")
            if not ranks:
                continue
            cells = []
            for bucket in ("1", "2", "3", "4-5", "6-10"):
                r = ranks.get(bucket)
                cells.append(
                    "—" if r is None else f"{pct(r['irr_full_cut'])} (n={r['base_irr_full']}) / {pct(r['used_full_lost'])} (n={r['base_used_full']})"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        census = block["demotion_census"]
        lines += [
            "",
            "### Demotion strength census (applicable event × marked-node cases)",
            "",
            "| window | events | cases | default m>0.9 | fc080_mw1 m>0.9 | fc080_mw1 m<=0.75 | fc080_mw1 median |",
            "|---|---|---|---|---|---|---|",
        ]
        for window in ("trainval", "holdout"):
            c = census[window]
            lines.append(
                f"| {window} | {c['_events']} | {c['_applicable']} | {pct(c['default']['share_gt_0_9'])} | {pct(c['fc080_mw1']['share_gt_0_9'])} "
                f"| {pct(c['fc080_mw1']['share_le_0_75'])} | {c['fc080_mw1']['p50']} |"
            )
        lines.append("")
    lines += [(OUT_DIR / "runs" / "report_notes.md").read_text()] if (OUT_DIR / "runs" / "report_notes.md").exists() else []
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    results: dict[str, Any] = {"prereg": "artifacts/recall-precision/prereg.md", "hosts": {}}
    for host in HOSTS:
        block = host_block(host)
        block["verdicts"] = verdicts(host, block)
        results["hosts"][host] = block
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (OUT_DIR / "report.md").write_text(render(results))
    for host in HOSTS:
        print(host, [(v["id"], v["verdict"]) for v in results["hosts"][host]["verdicts"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
