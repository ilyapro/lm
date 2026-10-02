#!/usr/bin/env python3
"""Build and verify a public v2, v3 or v4 verdict from one private paired recall run.

CLI:
  build --seal PRIVATE/seal.json --raw PRIVATE/paired-raw.json
        --run-receipt PRIVATE/run-receipt.json
        --candidate-receipt artifacts/recall-applicability/candidate-v2.json
        --output artifacts/recall-applicability/report-v2.json
        --markdown artifacts/recall-applicability/report-v2.md
  verify --report artifacts/recall-applicability/report-v2.json [--require-success]

The seal's files map contains snapshot, goldset, previous_goldset and
baseline_outcomes entries, each {path, sha256}. The private run receipt has
schema_version=1, started_at_utc, ended_at_utc, seal_sha256, runner_sha256,
runner_input={path,sha256}, raw={path,sha256},
candidate_receipt={path,sha256}, consumption={path,sha256},
consumption_identity, pre_src_tree, post_src_tree, pre_source_sha256 and
post_source_sha256 (maps of production path to hash), command (argv), and
arms={baseline:{depth,max_results,schema_trigger}, candidate:{same keys}}.
The private consumption record has schema_version=1, identity, seal_sha256,
runner_input_sha256, candidate_commit, started_at_utc, status='started'.
It must exist before measurement. Verification reads frozen evidence and opens
the v3 snapshot read-only for literal validation; it never invokes the runner.

Reports contain aggregate numbers and hashes, never queries, facts, IDs or raw
responses. Public historical report.json is pinned as failed provenance.
After integration removes the execution worktree, verification can read its
candidate receipt at the same repository-relative path in this checkout, with
the original hash still required. Recorded paths and sealed bytes are retained.
"""
from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts/recall_applicability_eval.py'
OLD = ROOT / 'artifacts/recall-applicability/report.json'
PREREG = ROOT / 'artifacts/recall-applicability/prereg-v2.md'
PREREG_V3 = ROOT / 'artifacts/recall-applicability/prereg-v3.md'
PREREG_V4 = ROOT / 'artifacts/recall-applicability/prereg-v4.md'
V2_REPORT = ROOT / 'artifacts/recall-applicability/report-v2.json'
# Published by the archived v3 preregistration at e7c0ac6808380ced1e5668a7c55a1f5ba9e928df.
V3_SEAL_SHA256 = 'f27268596fe4dc4a0f7c8a98dcf735eb01df9d235ed4ef8b6dcd0a8e2a932cca'
CATEGORIES = {'concrete','compound','instruction','correction','cross-project','absent-knowledge','large-group'}
SOURCE_FILES = ('src/living_memory/retrieval.py','src/living_memory/score_gate.py')
CANDIDATE_RECEIPT = Path('artifacts/recall-applicability/candidate-v2.json')
CANDIDATE_RECEIPT_V3 = Path('artifacts/recall-applicability/candidate-v3.json')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def bound(entry: dict, label: str, repository_path: Path | None = None) -> Path:
    p = Path(entry['path']).resolve()
    if not p.exists() and repository_path is not None:
        require(p.parts[-len(repository_path.parts):] == repository_path.parts,
                f'{label}: invalid repository-relative path')
        p = ROOT / repository_path
    require(p.is_file() and digest(p) == entry['sha256'], f'{label}: missing or edited file')
    return p


def runner_location_matches(recorded: str, receipt: dict, candidate_location: Path = CANDIDATE_RECEIPT) -> bool:
    path = Path(recorded).resolve()
    if path == RUNNER:
        return True
    # Both tracked files must identify the same historical checkout. Content
    # identity is checked separately even when that checkout no longer exists.
    old_root = path.parent.parent
    return (path == old_root / RUNNER.relative_to(ROOT)
            and Path(receipt['candidate_receipt']['path']).resolve() == old_root / candidate_location
            and (not path.exists() or path.is_file() and digest(path) == receipt['runner_sha256']))


def git(*args: str) -> str:
    return subprocess.check_output(['git','-C',str(ROOT),*args], text=True).strip()


def runner_cases(cases: list[dict]) -> dict:
    # Only the runner-required shape may change. No oracle value is inferred.
    return {'cases': [{**{k:c[k] for k in ('case_id','query','scope','depth','max_results','category')},
            'required_facts':[{'text':x,'acceptable_source_ids':c['acceptable_source_ids']}
                              for x in c['required_facts']]} for c in cases]}


def valid_indexes(row: dict, count: int) -> set[int]:
    idx = row['reached_fact_indexes']
    require(isinstance(idx,list) and all(type(i) is int and 0 <= i < count for i in idx)
            and idx == sorted(set(idx)), 'invalid reached fact indexes')
    require(row['required_fact_count'] == count, 'fact denominator mismatch')
    return set(idx)


def metric(row: dict, count: int) -> dict:
    indexes = valid_indexes(row,count)
    calls = row['calls']
    require(isinstance(calls,list) and calls and calls[0]['tool']=='memory_recall', 'invalid call chain')
    require(all(x['tool'] in ('memory_recall','memory_lookup') and
                type(x['response_bytes']) is int and x['response_bytes'] >= 0 and
                type(x['latency_ms']) in (int,float) and x['latency_ms'] >= 0 for x in calls), 'invalid calls')
    n = row['necessary_call_count']
    require(n is None or type(n) is int and 1 <= n <= len(calls), 'invalid necessary prefix')
    require(row['necessary_sequence'] == ([x['tool'] for x in calls[:n]] if n else None), 'necessary sequence mismatch')
    require(row['necessary_response_bytes'] == (sum(x['response_bytes'] for x in calls[:n]) if n else None), 'necessary bytes mismatch')
    require(row['call_count']==len(calls) and row['recall_count']==sum(x['tool']=='memory_recall' for x in calls)
            and row['lookup_count']==sum(x['tool']=='memory_lookup' for x in calls), 'call counts mismatch')
    require(row['recall_count']==1 and all(x['tool']=='memory_lookup' for x in calls[1:]), 'unexpected call order')
    require(row['response_bytes']==sum(x['response_bytes'] for x in calls), 'response bytes mismatch')
    require(abs(row['warm_latency_ms']-sum(x['latency_ms'] for x in calls)) <= .002, 'latency mismatch')
    ranks = row['relevant_ranks']+row['irrelevant_ranks']
    require(all(type(i) is int and 1 <= i <= 4 for i in ranks) and len(ranks)==len(set(ranks)), 'rank indexes invalid')
    require(row['irrelevant_top_four']==sum(i<=4 for i in row['irrelevant_ranks']), 'irrelevant slots mismatch')
    require(row['sufficient'] == (n is not None if count else len(ranks)==0), 'sufficiency mismatch')
    require((n is not None)==(len(indexes)==count) if count else n is None, 'necessary prefix/reachability mismatch')
    return {'required_facts':count,'reached_facts':len(indexes), 'sufficient_cases':int(row['sufficient']),
            'relevant_top_four':len(row['relevant_ranks']), 'irrelevant_top_four':row['irrelevant_top_four'],
            'recall_calls':row['recall_count'],'lookup_calls':row['lookup_count'],'total_calls':row['call_count'],
            'total_response_bytes':row['response_bytes'],'warm_latency_ms':row['warm_latency_ms'],
            'necessary_calls_reached_cases':n or 0,
            'necessary_response_bytes_reached_cases':row['necessary_response_bytes'] or 0,
            'exhausted_calls_unreached_cases':len(calls) if count and n is None else 0,
            'exhausted_response_bytes_unreached_cases':row['response_bytes'] if count and n is None else 0,
            'necessary_sequences':{' -> '.join(row['necessary_sequence']):1} if n else {},
            'exhausted_sequences':{' -> '.join(x['tool'] for x in calls):1} if count and n is None else {}}


def aggregate(rows: list[dict]) -> dict:
    result = {'cases':len(rows)}
    keys = ('required_facts','reached_facts','sufficient_cases','relevant_top_four','irrelevant_top_four',
            'recall_calls','lookup_calls','total_calls','total_response_bytes','necessary_calls_reached_cases',
            'necessary_response_bytes_reached_cases','exhausted_calls_unreached_cases','exhausted_response_bytes_unreached_cases')
    for key in keys:
        result[key]=sum(x[key] for x in rows)
    for key in ('necessary_sequences','exhausted_sequences'):
        counts=Counter()
        for row in rows: counts.update(row[key])
        result[key]=dict(sorted(counts.items()))
    result['warm_latency_ms_sum']=round(sum(x['warm_latency_ms'] for x in rows),3)
    result['warm_latency_ms_median']=round(statistics.median(x['warm_latency_ms'] for x in rows),3) if rows else 0
    return result



def production_delta() -> dict:
    changes={}
    for name in SOURCE_FILES:
        before=git('show','b9769d8:'+name).splitlines()
        after=(ROOT/name).read_text().splitlines()
        added=removed=0
        for tag,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes():
            if tag!='equal':
                removed+=b-a; added+=d-c
        changes[name]={'baseline_lines':len(before),'candidate_lines':len(after),
                       'added_lines':added,'removed_lines':removed}
    return changes


def seal_generation(seal: dict) -> int:
    generation = seal.get('generation', 2)
    require(type(generation) is int and generation in (2, 3, 4), 'unsupported seal generation')
    if generation == 2:
        require('prior_seals' not in seal, 'v2 seal cannot carry prior seals')
    if generation == 3:
        require(seal.get('schema_version') == 1 and
                set(seal.get('files', {})) == {'snapshot', 'goldset', 'previous_goldset', 'baseline_outcomes'} and
                isinstance(seal.get('prior_seals'), list) and len(seal['prior_seals']) == 1,
                'v3 seal schema')
        require(set(seal['prior_seals'][0]) == {'path','sha256'}, 'prior seal entry shape')
    if generation == 4:
        require(seal.get('schema_version') == 1 and seal.get('candidate_generation') == 3 and
                set(seal.get('files', {})) == {'snapshot', 'goldset', 'previous_goldset', 'baseline_outcomes'} and
                isinstance(seal.get('prior_seals'), list) and len(seal['prior_seals']) == 2 and
                all(set(entry) == {'path','sha256'} for entry in seal['prior_seals']),
                'v4 seal generation/schema')
    return generation


def check_baseline_sources(cases: list[dict], baseline: dict) -> None:
    """Source rank is an oracle preflight, separate from paired delivery."""
    by_id = {row['case_id']: row for row in baseline['outcomes']}
    protected = set()
    for case in cases:
        row = by_id[case['case_id']]
        ranked = row['ranked_ids']
        require(isinstance(ranked, list) and len(ranked) <= 4 and
                len(ranked) == len(set(ranked)) and all(isinstance(s, str) for s in ranked) and
                row['split'] == case['split'], 'baseline ranked ids invalid')
        expected = {source: ranked.index(source) + 1 for source in case['acceptable_source_ids'] if source in ranked}
        require(row['source_ranks'] == expected, 'baseline source ranks inconsistent')
        if case['split'] == 'holdout' and case['category'] in ('instruction','correction','cross-project') and expected:
            protected.add(case['category'])
    require(protected == {'instruction','correction','cross-project'}, 'protected baseline top-four source missing')


def validate_seal(seal_path: Path) -> tuple[dict, dict, list[dict]]:
    seal = read(seal_path)
    generation = seal_generation(seal)
    require(seal['schema_version'] == 1 and
            set(seal['files']) == {'snapshot', 'goldset', 'previous_goldset', 'baseline_outcomes'}, 'seal schema')
    files = {k: bound(v, k) for k, v in seal['files'].items()}
    old_hashes = read(OLD)['input_sha256']
    require(seal['files']['snapshot']['sha256'] == old_hashes['snapshot'] and
            seal['files']['previous_goldset']['sha256'] == old_hashes['goldset'], 'original frozen input mismatch')
    prereg = PREREG_V4 if generation == 4 else PREREG_V3 if generation == 3 else PREREG
    if generation == 4 or seal_path.resolve().is_relative_to('/home/sfx/p/ae/artifacts/recall-applicability'):
        require(prereg.is_file(), f'v{generation} preregistration missing')
    if prereg.is_file():
        prereg_text = prereg.read_text(encoding='utf-8')
        require(all(entry['sha256'] in prereg_text for entry in seal['files'].values()),
                'sealed hash absent from preregistration')
    gold = read(files['goldset']); previous = read(files['previous_goldset'])
    cases = gold['cases']; old_cases = previous['cases']
    if generation in (3, 4):
        prior_entry = seal['prior_seals'][0]
        prior_path = bound(prior_entry, 'prior v2 seal')
        prior = read(prior_path)
        require(seal_generation(prior) == 2 and prior.get('schema_version') == 1 and
                set(prior.get('files', {})) == set(seal['files']), 'prior v2 seal schema')
        prior_files = {k: bound(v, 'prior '+k) for k, v in prior['files'].items()}
        require(prior['files']['snapshot']['sha256'] == old_hashes['snapshot'] and
                prior['files']['previous_goldset']['sha256'] == old_hashes['goldset'],
                'prior original input mismatch')
        prior_baseline = read(prior_files['baseline_outcomes'])
        require(prior['baseline_commit'] == seal['baseline_commit'] and
                prior['baseline_src_tree'] == seal['baseline_src_tree'] and
                prior_baseline['code_head'] == prior['baseline_commit'] and
                prior_baseline['inputs']['snapshot_sha256'] == prior['files']['snapshot']['sha256'] and
                prior_baseline['inputs']['corpus_sha256'] == prior['files']['goldset']['sha256'],
                'prior baseline provenance mismatch')
        older_cases = old_cases + read(prior_files['goldset'])['cases']
        hold = [c for c in cases if c['split'] == 'holdout']
        require({c['query'] for c in hold}.isdisjoint(c['query'] for c in older_cases) and
                {c['topic_group'] for c in hold}.isdisjoint(c['topic_group'] for c in older_cases) and
                {s for c in hold for s in c['acceptable_source_ids']}.isdisjoint(
                    s for c in older_cases for s in c['acceptable_source_ids']),
                'prior population overlap')
        require([c for c in cases if c['split'] == 'development'] ==
                [c for c in read(prior_files['goldset'])['cases'] if c['split'] == 'development'],
                'development changed from v2')
        require(V2_REPORT.is_file() and read(V2_REPORT)['schema_version'] == 2 and
                read(V2_REPORT)['evidence']['seal']['sha256'] == digest(prior_path),
                'historical v2 report/seal mismatch')
        require(read(OLD)['verdict']['P1']['pass'] is False and
                read(V2_REPORT)['verdict']['P1']['pass'] is True,
                'historical verdict mismatch')
        if prereg.is_file():
            require(all(value in prereg_text for value in
                        (digest(OLD), digest(V2_REPORT), digest(prior_path))),
                    'historical hash absent from preregistration')
        if generation == 4:
            historical_entry = seal['prior_seals'][1]
            require(historical_entry['sha256'] == V3_SEAL_SHA256, 'published v3 seal hash mismatch')
            historical_path = bound(historical_entry, 'prior v3 seal')
            historical = read(historical_path)
            require(seal_generation(historical) == 3 and
                    historical['prior_seals'][0] == prior_entry and
                    historical['baseline_commit'] == seal['baseline_commit'] and
                    historical['baseline_src_tree'] == seal['baseline_src_tree'] and
                    set(historical['files']) == set(seal['files']), 'prior v3 lineage mismatch')
            historical_files = {k: bound(v, 'prior v3 '+k) for k, v in historical['files'].items()}
            require(historical['files']['snapshot'] == seal['files']['snapshot'] and
                    historical['files']['previous_goldset'] == seal['files']['previous_goldset'],
                    'prior v3 original input mismatch')
            historical_baseline = read(historical_files['baseline_outcomes'])
            historical_cases = read(historical_files['goldset'])['cases']
            require(historical_baseline['code_head'] == seal['baseline_commit'] and
                    historical_baseline['inputs']['snapshot_sha256'] == seal['files']['snapshot']['sha256'] and
                    historical_baseline['inputs']['corpus_sha256'] == historical['files']['goldset']['sha256'] and
                    len(historical_baseline['outcomes']) == len(historical_cases) and
                    {r['case_id'] for r in historical_baseline['outcomes']} == {c['case_id'] for c in historical_cases},
                    'prior v3 baseline provenance mismatch')
            require([c for c in historical_cases if c['split'] == 'development'] ==
                    [c for c in old_cases if c['split'] == 'development'], 'prior v3 development changed')
            older_cases += historical_cases
            require({c['query'] for c in hold}.isdisjoint(c['query'] for c in historical_cases) and
                    {c['topic_group'] for c in hold}.isdisjoint(c['topic_group'] for c in historical_cases) and
                    {s for c in hold for s in c['acceptable_source_ids']}.isdisjoint(
                        s for c in historical_cases for s in c['acceptable_source_ids']),
                    'prior v3 population overlap')
            if prereg.is_file():
                require(all(value in prereg_text for value in
                            (digest(historical_path),
                             *(entry['sha256'] for entry in historical['files'].values()),
                             *(entry['sha256'] for entry in prior['files'].values()))),
                        'historical hash absent from preregistration')
        baseline = read(files['baseline_outcomes'])
        require(seal['baseline_commit'] == 'b9769d84e3188ee1e646627ebe1b4d454f69d6f9' and
                seal['baseline_src_tree'] == git('rev-parse', seal['baseline_commit']+':src') and
                baseline['code_head'] == seal['baseline_commit'] and
                baseline['inputs']['snapshot_sha256'] == seal['files']['snapshot']['sha256'] and
                baseline['inputs']['corpus_sha256'] == seal['files']['goldset']['sha256'] and
                {r['case_id'] for r in baseline['outcomes']} == {c['case_id'] for c in cases} and
                len(baseline['outcomes']) == len(cases), 'baseline code provenance mismatch')
        if generation == 4:
            require(all(c['scope'] is None and c['depth'] == 1 and c['max_results'] == 4
                        for c in cases if c['split'] == 'holdout'), 'v4 broad recall parameters mismatch')
            check_baseline_sources(cases, baseline)
        # The frozen SQLite file is opened read-only by the existing literal-clause validator.
        from recall_applicability_corpus import validate
        validate(files['snapshot'], files['goldset'])
    return seal, files, cases

def validate_packet(seal_path: Path, raw_path: Path, receipt_path: Path, candidate_path: Path) -> tuple[dict,dict,dict,list[dict]]:
    seal, raw, receipt, candidate = map(read,(seal_path,raw_path,receipt_path,candidate_path))
    seal,files,cases=validate_seal(seal_path)
    generation=seal_generation(seal)
    candidate_location=CANDIDATE_RECEIPT_V3 if generation>=3 else CANDIDATE_RECEIPT
    if generation == 4:
        require(candidate_path.resolve().parts[-len(candidate_location.parts):] == candidate_location.parts and
                Path(receipt['candidate_receipt']['path']).resolve().parts[-len(candidate_location.parts):] == candidate_location.parts,
                'candidate-v3 receipt location mismatch')
    require(candidate['baseline_commit']==seal['baseline_commit'] and
            re.fullmatch(r'[0-9a-f]{40}',candidate['baseline_commit']) is not None and
            re.fullmatch(r'[0-9a-f]{40}',candidate['candidate_commit']) is not None and
            candidate['baseline_commit'].startswith('b9769d8'), 'baseline/candidate code mismatch')
    require(raw['baseline_commit']=='b9769d8' and raw['snapshot_sha256']==seal['files']['snapshot']['sha256'], 'raw provenance mismatch')
    require(candidate['src_tree']==git('rev-parse','HEAD:src'), 'current source tree mismatch')
    require(candidate['src_tree']==git('rev-parse',candidate['candidate_commit']+':src'), 'candidate commit/source tree mismatch')
    require(seal['baseline_src_tree']==git('rev-parse',seal['baseline_commit']+':src'), 'baseline source tree mismatch')
    require(set(candidate['source_sha256'])==set(SOURCE_FILES), 'source receipt shape')
    for name in SOURCE_FILES:
        require(digest(ROOT/name)==candidate['source_sha256'][name], f'current source hash mismatch: {name}')
    require(receipt['schema_version']==1 and receipt['seal_sha256']==digest(seal_path), 'run seal mismatch')
    require(receipt['runner_sha256']==digest(RUNNER), 'runner hash mismatch')
    require(bound(receipt['raw'],'raw')==raw_path.resolve() and
            bound(receipt['candidate_receipt'],'candidate receipt',candidate_location)==candidate_path.resolve(), 'run file mismatch')
    input_path=bound(receipt['runner_input'],'runner input')
    consumption_path=bound(receipt['consumption'],'consumption')
    consumption=read(consumption_path)
    require(consumption['schema_version']==1 and consumption['status']=='started' and
            consumption['identity']==receipt['consumption_identity'] and
            consumption['seal_sha256']==digest(seal_path) and
            consumption['runner_input_sha256']==digest(input_path) and
            consumption['candidate_commit']==candidate['candidate_commit'] and
            consumption['started_at_utc']==receipt['started_at_utc'], 'single consumption identity mismatch')
    require(receipt['started_at_utc'] < receipt['ended_at_utc'], 'run timestamps')
    require(receipt['pre_src_tree']==receipt['post_src_tree']==candidate['src_tree'] and
            receipt['pre_source_sha256']==receipt['post_source_sha256']==candidate['source_sha256'], 'pre/post source mismatch')
    require(receipt['arms']=={'baseline':{'depth':1,'max_results':4,'schema_trigger':'default'},
                              'candidate':{'depth':1,'max_results':4,'schema_trigger':'default'}}, 'arm parameters or forbidden name mode')
    require(raw['arm_order']=='alternating by case index' and 'fresh process and DB per case and arm' in raw['warmup'], 'isolation mismatch')
    require(raw['environment']=='inherited identically; PYTHONPATH differs only by code root', 'environment mismatch')
    cmd=receipt['command']
    require(isinstance(cmd,list) and len(cmd)>=2 and all(isinstance(x,str) for x in cmd) and
            not any('LM_RECALL_SCHEMA_TRIGGER=name' in x for x in cmd) and
            '--worker' not in cmd and runner_location_matches(cmd[1],receipt,candidate_location) and
            '--snapshot' in cmd and '--cases' in cmd and '--out' in cmd and
            Path(cmd[cmd.index('--snapshot')+1]).resolve()==files['snapshot'] and
            Path(cmd[cmd.index('--cases')+1]).resolve()==input_path and
            Path(cmd[cmd.index('--out')+1]).resolve()==raw_path.resolve(), 'run command mismatch')
    gold=read(files['goldset']); previous=read(files['previous_goldset'])
    cases=gold['cases']; old_cases=previous['cases']
    require(gold['schema_version']==1 and len(cases)>=16 and len({c['case_id'] for c in cases})==len(cases)
            and all(c['split'] in ('development','holdout') and isinstance(c['topic_group'],str)
                    and bool(c['topic_group']) and isinstance(c['query'],str) and bool(c['query'].strip())
                    and isinstance(c['required_facts'],list) and
                    all(isinstance(x,str) and bool(x) for x in c['required_facts']) for c in cases), 'corpus shape')
    require(read(input_path)==runner_cases(cases), 'runner input is not exact shape-only conversion')
    require(len({c['query'] for c in cases})==len(cases), 'duplicate query')
    dev=[c for c in cases if c['split']=='development']; hold=[c for c in cases if c['split']=='holdout']
    require(len(hold)>=12 and dev and CATEGORIES <= {c['category'] for c in hold}, 'holdout coverage')
    require({c['topic_group'] for c in dev}.isdisjoint(c['topic_group'] for c in hold), 'split groups overlap')
    require({s for c in dev for s in c['acceptable_source_ids']}.isdisjoint(s for c in hold for s in c['acceptable_source_ids']), 'split sources overlap')
    require(len({s for c in hold for s in c['acceptable_source_ids']})==sum(len(c['acceptable_source_ids']) for c in hold), 'holdout sources reused')
    require({c['query'] for c in hold}.isdisjoint(c['query'] for c in old_cases) and
            {c['topic_group'] for c in hold}.isdisjoint(c['topic_group'] for c in old_cases) and
            {s for c in hold for s in c['acceptable_source_ids']}.isdisjoint(s for c in old_cases for s in c['acceptable_source_ids']), 'old/new holdout overlap')
    require([c for c in cases if c['split']=='development']==[c for c in old_cases if c['split']=='development'], 'development changed')
    baseline=read(files['baseline_outcomes'])
    require(baseline['schema_version']==1 and baseline['inputs']['snapshot_sha256']==seal['files']['snapshot']['sha256'] and
            baseline['inputs']['corpus_sha256']==seal['files']['goldset']['sha256'] and
            baseline['code_head']==seal['baseline_commit'] and
            len(baseline['outcomes'])==len(cases) and
            {r['case_id'] for r in baseline['outcomes']}=={c['case_id'] for c in cases}, 'baseline receipt mismatch')
    require(set(raw['cases'])=={c['case_id'] for c in cases}, 'raw case population mismatch')
    require(raw['production_code_delta']==production_delta(), 'production complexity mismatch')
    for c in cases:
        require(c['depth']==1 and c['max_results']==4 and c['scope'] is None or
                c['depth']==1 and c['max_results']==4 and isinstance(c['scope'],str), 'recall parameters mismatch')
        require(bool(c['required_facts']) != (c['category']=='absent-knowledge'), 'absent case answer key')
        require(bool(c['acceptable_source_ids']) == bool(c['required_facts']), 'source denominator')
    require(OLD.is_file() and read(OLD)['verdict']['P1']['pass'] is False, 'historical failure missing')
    return seal,raw,receipt,cases


def build_data(seal_path: Path, raw_path: Path, receipt_path: Path, candidate_path: Path) -> dict:
    seal,raw,receipt,cases=validate_packet(seal_path,raw_path,receipt_path,candidate_path)
    by_split=defaultdict(list); by_category=defaultdict(lambda:defaultdict(list)); groups=defaultdict(list)
    losses=Counter(); misses=Counter(); dev_losses=Counter(); dev_misses=Counter()
    improved=set(); loss_details=Counter(); protected=Counter()
    for c in cases:
        arms=raw['cases'][c['case_id']]; require(set(arms)=={'baseline','candidate'}, 'arm population')
        n=len(c['required_facts']); measures={}
        for arm in ('baseline','candidate'):
            row=arms[arm]
            require(row['case_id']==c['case_id'] and row['category']==c['category'], 'case identity mismatch')
            measures[arm]=metric(row,n)
            by_split[(c['split'],arm)].append(measures[arm]); by_category[(c['split'],c['category'])][arm].append(measures[arm])
        base=valid_indexes(arms['baseline'],n); cand=valid_indexes(arms['candidate'],n)
        lost=sorted(base-cand)
        require(arms['candidate']['lost_fact_indexes']==lost, 'runner fact loss mismatch')
        if c['split']=='holdout':
            losses[c['category']]+=len(lost); misses[c['category']]+=n-len(base)
            if c['category'] in ('instruction','correction','cross-project'):
                protected[c['category']]+=len(base)
            groups[c['topic_group']].append((measures['baseline']['irrelevant_top_four'],measures['candidate']['irrelevant_top_four']))
            if lost: loss_details[c['category']]+=1
        else:
            dev_losses[c['category']]+=len(lost); dev_misses[c['category']]+=n-len(base)
    for group, pairs in groups.items():
        if sum(a for a,b in pairs)>sum(b for a,b in pairs): improved.add(group)
    generation=seal_generation(seal)
    p1=(len(improved)>=2 and sum(losses.values())==0 and sum(dev_losses.values())==0 and
        all(protected[c]>0 for c in ('instruction','correction','cross-project')))
    report={'schema_version':generation,'evidence':{'seal':{'path':str(seal_path.resolve()),'sha256':digest(seal_path)},
            'raw':{'path':str(raw_path.resolve()),'sha256':digest(raw_path)},
            'run_receipt':{'path':str(receipt_path.resolve()),'sha256':digest(receipt_path)},
            'candidate_receipt':receipt['candidate_receipt'],
            'historical_failed_report_sha256':digest(OLD),'runner_sha256':digest(RUNNER),
            'snapshot_sha256':seal['files']['snapshot']['sha256'],'baseline_commit':seal['baseline_commit'],
            'candidate_commit':read(candidate_path)['candidate_commit'],'src_tree':read(candidate_path)['src_tree'],
            'consumption_identity':receipt['consumption_identity']},
            'results':{split:{'overall':{arm:aggregate(by_split[(split,arm)]) for arm in ('baseline','candidate')},
                      'categories':{cat:{arm:aggregate(rows[arm]) for arm in ('baseline','candidate')}
                                    for (s,cat),rows in sorted(by_category.items()) if s==split}}
                       for split in ('development','holdout')},
            'holdout':{'case_count':sum(c['split']=='holdout' for c in cases),
                       'topic_group_count':len(groups),'improved_topic_group_count':len(improved),
                       'baseline_missed_facts':dict(sorted(misses.items())),
                       'new_fact_losses':dict(sorted(losses.items())),
                       'loss_cases':dict(sorted(loss_details.items()))},
            'production_code_delta':raw['production_code_delta'],
            'verdict':{'P1':{'pass':p1,'threshold':'at least two distinct improved holdout topic groups and zero baseline-reachable fact losses'},
                       'P2':{'pass':True,'basis':'validated sealed provenance, paired parameters and aggregate arithmetic'},
                       'P4':{'pass':False,'basis':'root full-suite check is external to this report'}}}
    if generation>=3:
        report['evidence']['historical_v2_report_sha256']=digest(V2_REPORT)
        report['evidence']['prior_seal_sha256']=seal['prior_seals'][0]['sha256']
        report['development']={'baseline_missed_facts':dict(sorted(dev_misses.items())),
                               'new_fact_losses':dict(sorted(dev_losses.items()))}
        report['holdout']['baseline_reached_protected_facts']=dict(sorted(protected.items()))
        report['verdict']['P1']['threshold'] += '; non-vacuous instruction, correction and cross-project controls; zero development losses'
    if generation==4:
        report['evidence']['prior_v3_seal_sha256']=seal['prior_seals'][1]['sha256']
        report['verdict']['P1']['threshold'] += '; broad-scope top-four baseline source controls'
    return report


def verify(path: Path, success: bool=False) -> dict:
    report=read(path)
    e=report['evidence']
    generation=seal_generation(read(bound(e['seal'],'seal')))
    candidate_location=CANDIDATE_RECEIPT_V3 if generation>=3 else CANDIDATE_RECEIPT
    require(report==build_data(bound(e['seal'],'seal'),bound(e['raw'],'raw'),bound(e['run_receipt'],'run receipt'),
                               bound(e['candidate_receipt'],'candidate receipt',candidate_location)), 'forged or stale aggregate report')
    gold=read(bound(read(Path(e['seal']['path']))['files']['goldset'],'goldset'))
    public=path.read_text(encoding='utf-8')
    private=[value for case in gold['cases'] for value in
             [case['query'],*case['required_facts'],*case['acceptable_source_ids']]]
    require(not any(value and value in public for value in private), 'private content leaked into report')
    if success: require(report['verdict']['P1']['pass'], 'P1 threshold unmet')
    return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build',help='build validated public aggregate from private frozen evidence')
    for name in ('seal','raw','run-receipt','candidate-receipt','output','markdown'):
        b.add_argument('--'+name,type=Path,required=True)
    v=sub.add_parser('verify',help='recompute and validate report without rerunning holdout')
    v.add_argument('--report',type=Path,required=True); v.add_argument('--require-success',action='store_true')
    s=sub.add_parser('validate-seal',help='check a sealed packet before one-shot measurement')
    s.add_argument('--seal',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.command=='build':
            report=build_data(a.seal,a.raw,a.run_receipt,a.candidate_receipt)
            a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            h=report['holdout']
            lines=[f"# Recall applicability v{report['schema_version']}",'',
                f"P1: {'PASS' if report['verdict']['P1']['pass'] else 'FAIL'}; "
                f"{h['improved_topic_group_count']} improved independent groups; "
                f"{sum(h['new_fact_losses'].values())} new fact losses; "
                f"{sum(h['baseline_missed_facts'].values())} baseline misses.",'',
                '| Holdout category | Baseline reached / required | Candidate reached / required | Irrelevant top four (baseline → candidate) | New losses | Baseline misses |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
            for category, arms in report['results']['holdout']['categories'].items():
                base,cand=arms['baseline'],arms['candidate']
                lines.append(f"| {category} | {base['reached_facts']} / {base['required_facts']} | "
                    f"{cand['reached_facts']} / {cand['required_facts']} | "
                    f"{base['irrelevant_top_four']} → {cand['irrelevant_top_four']} | "
                    f"{h['new_fact_losses'].get(category,0)} | {h['baseline_missed_facts'].get(category,0)} |")
            lines.extend(['','Per-category call, lookup, necessary prefix, exhausted chain, byte, latency and production complexity aggregates are in the JSON report.',
                          'P4 full-suite success is established by the parent, not this report.',''])
            if report['schema_version']>=3:
                lines.extend([f"Development new fact losses: {sum(report['development']['new_fact_losses'].values())}; "
                              f"baseline misses: {sum(report['development']['baseline_missed_facts'].values())}.",
                              'Protected baseline-reached facts: '+', '.join(
                                  f'{key}={value}' for key,value in h['baseline_reached_protected_facts'].items()),''])
            a.markdown.write_text('\n'.join(lines),encoding='utf-8')
        elif a.command=='validate-seal': validate_seal(a.seal)
        else: verify(a.report,a.require_success)
    except (ValueError,KeyError,TypeError,IndexError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        p.exit(1,f'verdict: {exc}\n')

if __name__=='__main__': main()
