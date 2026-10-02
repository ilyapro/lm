"""Synthetic fourth-generation corpus and candidate-v3 compatibility checks."""
from __future__ import annotations

import copy
import json
import sqlite3
import subprocess
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import recall_applicability_verdict as verdict
from test_recall_applicability_verdict import v3_packet, row, write


def baseline_rows(cases):
    return [{'case_id': c['case_id'], 'split': c['split'],
             'ranked_ids': c['acceptable_source_ids'][:1],
             'source_ranks': {s: 1 for s in c['acceptable_source_ids'][:1]}}
            for c in cases]


def reseal(p, path):
    seal = verdict.read(path)
    seal['files']['snapshot']['sha256'] = verdict.digest(p['snapshot'])
    base_path = Path(seal['files']['baseline_outcomes']['path'])
    baseline = verdict.read(base_path)
    baseline['inputs']['snapshot_sha256'] = verdict.digest(p['snapshot'])
    baseline['inputs']['corpus_sha256'] = seal['files']['goldset']['sha256']
    baseline['outcomes'] = baseline_rows(verdict.read(Path(seal['files']['goldset']['path']))['cases'])
    write(base_path, baseline)
    seal['files']['baseline_outcomes']['sha256'] = verdict.digest(base_path)
    write(path, seal)


def refresh_current(p):
    seal = verdict.read(p['seal'])
    seal['files']['goldset']['sha256'] = verdict.digest(p['gold'])
    baseline = verdict.read(p['baseline'])
    baseline['inputs']['snapshot_sha256'] = verdict.digest(p['snapshot'])
    baseline['inputs']['corpus_sha256'] = verdict.digest(p['gold'])
    baseline['outcomes'] = baseline_rows(verdict.read(p['gold'])['cases'])
    write(p['baseline'], baseline)
    seal['files']['baseline_outcomes']['sha256'] = verdict.digest(p['baseline'])
    write(p['seal'], seal)
    cases = verdict.read(p['gold'])['cases']
    write(p['input'], verdict.runner_cases(cases))
    raw = verdict.read(p['raw'])
    raw['snapshot_sha256'] = verdict.digest(p['snapshot'])
    raw['cases'] = {}
    for c in cases:
        b = row(c, 'baseline')
        a = row(c, 'candidate', irrelevant=2 if c['case_id'] in ('h0', 'h1') else 3)
        a['lost_fact_indexes'] = []
        raw['cases'][c['case_id']] = {'baseline': b, 'candidate': a}
    write(p['raw'], raw)
    consumption = verdict.read(p['consume'])
    consumption['seal_sha256'] = verdict.digest(p['seal'])
    consumption['runner_input_sha256'] = verdict.digest(p['input'])
    write(p['consume'], consumption)
    receipt = verdict.read(p['receipt'])
    receipt['seal_sha256'] = verdict.digest(p['seal'])
    for key, path in [('raw', p['raw']), ('runner_input', p['input']), ('consumption', p['consume'])]:
        receipt[key]['sha256'] = verdict.digest(path)
    write(p['receipt'], receipt)
    seal = verdict.read(p['seal'])
    v2 = verdict.read(p['prior_v2'])
    v3 = verdict.read(p['prior_v3'])
    verdict.PREREG_V4.write_text('\n'.join([
        *(x['sha256'] for x in seal['files'].values()),
        *(x['sha256'] for x in v2['files'].values()),
        *(x['sha256'] for x in v3['files'].values()),
        verdict.digest(p['prior_v2']), verdict.digest(p['prior_v3']),
        verdict.digest(verdict.OLD), verdict.digest(verdict.V2_REPORT)]))


@pytest.fixture
def p(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict, 'OLD', tmp_path/'old-report.json')
    monkeypatch.setattr(verdict, 'PREREG', tmp_path/'prereg-v2.md')
    monkeypatch.setattr(verdict, 'PREREG_V3', tmp_path/'prereg-v3.md')
    monkeypatch.setattr(verdict, 'V2_REPORT', tmp_path/'report-v2.json')
    q = v3_packet(tmp_path, monkeypatch)
    monkeypatch.setattr(verdict, 'PREREG_V4', tmp_path/'prereg-v4.md')
    q['snapshot'] = tmp_path/'snapshot.sqlite3'
    q['baseline'] = tmp_path/'baseline.json'
    q['prior_v2'] = q['prior']
    q['prior_v3'] = tmp_path/'seal-v3.json'
    q['prior_v3'].write_bytes(q['seal'].read_bytes())
    v3 = verdict.read(q['prior_v3'])
    v3_gold = tmp_path/'gold-v3.json'
    v3_gold.write_bytes(q['gold'].read_bytes())
    v3['files']['goldset'] = {'path': str(v3_gold), 'sha256': verdict.digest(v3_gold)}
    v3_base = tmp_path/'baseline-v3.json'
    v3_base.write_bytes(q['baseline'].read_bytes())
    v3['files']['baseline_outcomes'] = {'path': str(v3_base), 'sha256': verdict.digest(v3_base)}
    write(q['prior_v3'], v3)
    gold = verdict.read(q['gold'])
    for c in gold['cases']:
        if c['split'] == 'holdout':
            c['query'] = 'fourth ' + c['query']
            c['topic_group'] = 'fourth ' + c['topic_group']
            c['acceptable_source_ids'] = ['fourth ' + s for s in c['acceptable_source_ids']]
    write(q['gold'], gold)
    with sqlite3.connect(q['snapshot']) as db:
        for c in gold['cases']:
            for source in c['acceptable_source_ids']:
                db.execute('INSERT OR IGNORE INTO nodes VALUES (?,?)', (source, ' '.join(c['required_facts'])))
    old = verdict.read(verdict.OLD)
    old['input_sha256']['snapshot'] = verdict.digest(q['snapshot'])
    write(verdict.OLD, old)
    reseal(q, q['prior_v2'])
    write(verdict.V2_REPORT, {'schema_version': 2, 'evidence': {'seal': {'sha256': verdict.digest(q['prior_v2'])}},
                              'verdict': {'P1': {'pass': True}}})
    v3 = verdict.read(q['prior_v3'])
    v3['prior_seals'] = [{'path': str(q['prior_v2']), 'sha256': verdict.digest(q['prior_v2'])}]
    write(q['prior_v3'], v3)
    reseal(q, q['prior_v3'])
    # The historical packet is intact, but its cross-project source control
    # was inadequate. It remains an exclusion population.
    v3_baseline = verdict.read(v3_base)
    v3_cases = verdict.read(v3_gold)['cases']
    inadequate = {c['case_id'] for c in v3_cases if c['split'] == 'holdout' and c['category'] == 'cross-project'}
    for outcome in v3_baseline['outcomes']:
        if outcome['case_id'] in inadequate:
            outcome['ranked_ids'] = []
            outcome['source_ranks'] = {}
    write(v3_base, v3_baseline)
    v3 = verdict.read(q['prior_v3'])
    v3['files']['baseline_outcomes']['sha256'] = verdict.digest(v3_base)
    write(q['prior_v3'], v3)
    monkeypatch.setattr(verdict, 'V3_SEAL_SHA256', verdict.digest(q['prior_v3']))
    seal = verdict.read(q['seal'])
    seal['generation'] = 4
    seal['candidate_generation'] = 3
    seal['prior_seals'] = [
        {'path': str(q['prior_v2']), 'sha256': verdict.digest(q['prior_v2'])},
        {'path': str(q['prior_v3']), 'sha256': verdict.digest(q['prior_v3'])}]
    seal['files']['snapshot']['sha256'] = verdict.digest(q['snapshot'])
    write(q['seal'], seal)
    candidate = tmp_path/'artifacts/recall-applicability/candidate-v3.json'
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(q['candidate'].read_bytes())
    q['candidate'] = candidate
    receipt = verdict.read(q['receipt'])
    receipt['candidate_receipt'] = {'path': str(candidate), 'sha256': verdict.digest(candidate)}
    write(q['receipt'], receipt)
    refresh_current(q)
    return q


def build(p):
    return verdict.build_data(p['seal'], p['raw'], p['receipt'], p['candidate'])


def test_valid_v4_and_inadequate_historical_v3(p, tmp_path):
    old_v3 = verdict.read(Path(verdict.read(p['prior_v3'])['files']['baseline_outcomes']['path']))
    assert all(not r['source_ranks'] for r in old_v3['outcomes'] if
               r['case_id'] in {c['case_id'] for c in verdict.read(
                   Path(verdict.read(p['prior_v3'])['files']['goldset']['path']))['cases']
                   if c['split'] == 'holdout' and c['category'] == 'cross-project'})
    assert verdict.validate_seal(p['seal'])[0]['candidate_generation'] == 3
    result = build(p)
    assert result['schema_version'] == 4 and result['verdict']['P1']['pass']
    report = write(tmp_path/'report.json', result)
    verdict.verify(report, True)
    assert result['evidence']['prior_v3_seal_sha256'] == verdict.digest(p['prior_v3'])


def test_v4_cli_interfaces(p, tmp_path, monkeypatch):
    out = tmp_path/'aggregate.json'
    markdown = tmp_path/'aggregate.md'
    monkeypatch.setattr(sys, 'argv', ['verdict', 'validate-seal', '--seal', str(p['seal'])])
    verdict.main()
    monkeypatch.setattr(sys, 'argv', ['verdict', 'build', '--seal', str(p['seal']), '--raw', str(p['raw']),
        '--run-receipt', str(p['receipt']), '--candidate-receipt', str(p['candidate']),
        '--output', str(out), '--markdown', str(markdown)])
    verdict.main()
    assert markdown.is_file() and verdict.read(out)['schema_version'] == 4
    monkeypatch.setattr(sys, 'argv', ['verdict', 'verify', '--report', str(out), '--require-success'])
    verdict.main()


def test_original_development_must_remain_unchanged(p):
    gold = verdict.read(p['gold'])
    gold['cases'][0]['query'] += ' changed'
    write(p['gold'], gold)
    refresh_current(p)
    with pytest.raises(ValueError, match='development changed'):
        verdict.validate_seal(p['seal'])


@pytest.mark.parametrize('population', ['original', 'v2', 'v3'])
@pytest.mark.parametrize('field', ['query', 'topic_group', 'acceptable_source_ids'])
def test_overlap_with_each_population(p, population, field):
    older = (Path(verdict.read(p['prior_v2'])['files']['goldset']['path']) if population == 'v2' else
             Path(verdict.read(p['prior_v3'])['files']['goldset']['path']) if population == 'v3' else
             Path(verdict.read(p['seal'])['files']['previous_goldset']['path']))
    gold = verdict.read(p['gold'])
    current = next(c for c in gold['cases'] if c['split'] == 'holdout' and c['category'] == 'concrete')
    previous = next(c for c in verdict.read(older)['cases'] if c['split'] == 'holdout' and c['category'] == 'concrete')
    current[field] = copy.deepcopy(previous[field])
    write(p['gold'], gold)
    refresh_current(p)
    with pytest.raises(ValueError, match='overlap'):
        verdict.validate_seal(p['seal'])


@pytest.mark.parametrize('mutation', ['omit', 'edit-v2', 'edit-v3', 'lineage', 'historical-report', 'unpublished-v3'])
def test_missing_or_edited_ancestry(p, mutation):
    seal = verdict.read(p['seal'])
    if mutation == 'omit':
        seal['prior_seals'].pop()
        write(p['seal'], seal)
    elif mutation.startswith('edit'):
        path = p['prior_v2'] if mutation == 'edit-v2' else p['prior_v3']
        path.write_text(path.read_text() + ' ')
    elif mutation == 'lineage':
        v3 = verdict.read(p['prior_v3'])
        v3['prior_seals'][0]['sha256'] = '0' * 64
        write(p['prior_v3'], v3)
        seal['prior_seals'][1]['sha256'] = verdict.digest(p['prior_v3'])
        write(p['seal'], seal)
        refresh_current(p)
    elif mutation == 'historical-report':
        verdict.V2_REPORT.write_text('{}')
    else:
        seal['prior_seals'][1]['sha256'] = '0' * 64
        write(p['seal'], seal)
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        verdict.validate_seal(p['seal'])


def test_narrowed_scope_and_rank_inconsistency(p):
    gold = verdict.read(p['gold'])
    gold['cases'][4]['scope'] = 'project:one'
    write(p['gold'], gold)
    refresh_current(p)
    with pytest.raises(ValueError, match='broad'):
        verdict.validate_seal(p['seal'])


def test_inconsistent_source_rank(p):
    baseline = verdict.read(p['baseline'])
    baseline['outcomes'][4]['source_ranks'] = {'forged-source': 1}
    write(p['baseline'], baseline)
    seal = verdict.read(p['seal'])
    seal['files']['baseline_outcomes']['sha256'] = verdict.digest(p['baseline'])
    write(p['seal'], seal)
    verdict.PREREG_V4.write_text(verdict.PREREG_V4.read_text() + '\n' + verdict.digest(p['baseline']))
    with pytest.raises(ValueError, match='source ranks'):
        verdict.validate_seal(p['seal'])


def test_missing_v4_public_receipt(p):
    verdict.PREREG_V4.unlink()
    with pytest.raises(ValueError, match='preregistration missing'):
        verdict.validate_seal(p['seal'])


@pytest.mark.parametrize('category', ['instruction', 'correction', 'cross-project'])
def test_missing_baseline_source_control(p, category):
    baseline = verdict.read(p['baseline'])
    case = next(c for c in verdict.read(p['gold'])['cases'] if c['split'] == 'holdout' and c['category'] == category)
    for r in baseline['outcomes']:
        if r['case_id'] == case['case_id']:
            r['ranked_ids'] = []
            r['source_ranks'] = {}
    write(p['baseline'], baseline)
    seal = verdict.read(p['seal'])
    seal['files']['baseline_outcomes']['sha256'] = verdict.digest(p['baseline'])
    write(p['seal'], seal)
    refresh_current(p)
    # Each protected category has two cases; clear both to remove the control.
    baseline = verdict.read(p['baseline'])
    ids = {c['case_id'] for c in verdict.read(p['gold'])['cases'] if c['split'] == 'holdout' and c['category'] == category}
    for r in baseline['outcomes']:
        if r['case_id'] in ids:
            r['ranked_ids'] = []
            r['source_ranks'] = {}
    write(p['baseline'], baseline)
    seal = verdict.read(p['seal'])
    seal['files']['baseline_outcomes']['sha256'] = verdict.digest(p['baseline'])
    write(p['seal'], seal)
    verdict.PREREG_V4.write_text(verdict.PREREG_V4.read_text() + '\n' + verdict.digest(p['baseline']))
    with pytest.raises(ValueError, match='protected baseline'):
        verdict.validate_seal(p['seal'])


@pytest.mark.parametrize('case_id', ['d0', 'h2', 'h3', 'h4'])
def test_fact_losses(p, case_id):
    raw = verdict.read(p['raw'])
    case = next(c for c in verdict.read(p['gold'])['cases'] if c['case_id'] == case_id)
    raw['cases'][case_id]['candidate'] = row(case, 'candidate', found=False)
    raw['cases'][case_id]['candidate']['lost_fact_indexes'] = [0]
    write(p['raw'], raw)
    receipt = verdict.read(p['receipt'])
    receipt['raw']['sha256'] = verdict.digest(p['raw'])
    write(p['receipt'], receipt)
    result = build(p)
    assert result['verdict']['P1']['pass'] is False


def test_insufficient_groups_forgery_and_lookup(p, tmp_path):
    raw = verdict.read(p['raw'])
    case = next(c for c in verdict.read(p['gold'])['cases'] if c['case_id'] == 'h1')
    raw['cases']['h1']['candidate'] = row(case, 'candidate')
    raw['cases']['h1']['candidate']['lost_fact_indexes'] = []
    write(p['raw'], raw)
    receipt = verdict.read(p['receipt'])
    receipt['raw']['sha256'] = verdict.digest(p['raw'])
    write(p['receipt'], receipt)
    result = build(p)
    assert not result['verdict']['P1']['pass']
    path = write(tmp_path/'report.json', result)
    verdict.verify(path)
    with pytest.raises(ValueError, match='threshold'):
        verdict.verify(path, True)
    result['verdict']['P1']['pass'] = True
    write(path, result)
    with pytest.raises(ValueError, match='forged'):
        verdict.verify(path)


@pytest.mark.parametrize('category', ['absent-knowledge', 'large-group'])
def test_missing_category(p, category):
    gold = verdict.read(p['gold'])
    for c in gold['cases']:
        if c['split'] == 'holdout' and c['category'] == category:
            c['category'] = 'concrete'
    write(p['gold'], gold)
    refresh_current(p)
    with pytest.raises(ValueError, match='category'):
        verdict.validate_seal(p['seal'])


def test_wrong_generation_candidate_location_and_stale_hashes(p):
    seal = verdict.read(p['seal'])
    seal['candidate_generation'] = 4
    write(p['seal'], seal)
    with pytest.raises(ValueError, match='generation'):
        verdict.validate_seal(p['seal'])
    seal['candidate_generation'] = 3
    write(p['seal'], seal)
    refresh_current(p)
    receipt = verdict.read(p['receipt'])
    receipt['runner_sha256'] = '0' * 64
    write(p['receipt'], receipt)
    with pytest.raises(ValueError, match='runner hash'):
        build(p)


def test_receipt_relocation_and_stale_candidate_code(p, tmp_path):
    candidate = verdict.read(p['candidate'])
    candidate['src_tree'] = '0' * 40
    write(p['candidate'], candidate)
    receipt = verdict.read(p['receipt'])
    receipt['candidate_receipt']['sha256'] = verdict.digest(p['candidate'])
    write(p['receipt'], receipt)
    with pytest.raises(ValueError, match='current source tree'):
        build(p)
    candidate['src_tree'] = verdict.git('rev-parse', 'HEAD:src')
    write(p['candidate'], candidate)
    receipt['candidate_receipt']['sha256'] = verdict.digest(p['candidate'])
    receipt['candidate_receipt']['path'] = str(tmp_path/'candidate-v4.json')
    write(p['receipt'], receipt)
    with pytest.raises(ValueError, match='candidate-v3 receipt location'):
        build(p)


def test_stale_source_hash_and_candidate_commit(p):
    candidate = verdict.read(p['candidate'])
    candidate['source_sha256'][verdict.SOURCE_FILES[0]] = '0' * 64
    write(p['candidate'], candidate)
    receipt = verdict.read(p['receipt'])
    receipt['candidate_receipt']['sha256'] = verdict.digest(p['candidate'])
    write(p['receipt'], receipt)
    with pytest.raises(ValueError, match='current source hash'):
        build(p)
    candidate['source_sha256'][verdict.SOURCE_FILES[0]] = verdict.digest(ROOT/verdict.SOURCE_FILES[0])
    candidate['candidate_commit'] = '0' * 40
    write(p['candidate'], candidate)
    receipt['candidate_receipt']['sha256'] = verdict.digest(p['candidate'])
    write(p['receipt'], receipt)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        build(p)


def test_necessary_lookup_recovers_protected_fact(p):
    raw = verdict.read(p['raw'])
    case = next(c for c in verdict.read(p['gold'])['cases'] if c['case_id'] == 'h2')
    raw['cases']['h2']['candidate'] = row(case, 'candidate', lookup=True)
    raw['cases']['h2']['candidate']['lost_fact_indexes'] = []
    write(p['raw'], raw)
    receipt = verdict.read(p['receipt'])
    receipt['raw']['sha256'] = verdict.digest(p['raw'])
    write(p['receipt'], receipt)
    result = build(p)
    assert result['verdict']['P1']['pass']
    assert result['results']['holdout']['categories']['instruction']['candidate']['lookup_calls'] > 0


def test_removed_worktree_uses_hash_bound_candidate_v3_receipt(p, tmp_path, monkeypatch):
    actual_root = verdict.ROOT
    mirror = tmp_path/'repository'
    (mirror/'artifacts/recall-applicability').mkdir(parents=True)
    (mirror/'scripts').mkdir()
    (mirror/'src').symlink_to(actual_root/'src', target_is_directory=True)
    (mirror/verdict.CANDIDATE_RECEIPT_V3).write_bytes(p['candidate'].read_bytes())
    mirrored_runner = mirror/'scripts/recall_applicability_eval.py'
    mirrored_runner.write_bytes(verdict.RUNNER.read_bytes())
    old_root = tmp_path/'removed-worktree'
    candidate = old_root/verdict.CANDIDATE_RECEIPT_V3
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(p['candidate'].read_bytes())
    runner = old_root/'scripts/recall_applicability_eval.py'
    runner.parent.mkdir()
    runner.write_bytes(verdict.RUNNER.read_bytes())
    receipt = verdict.read(p['receipt'])
    receipt['candidate_receipt'] = {'path': str(candidate), 'sha256': verdict.digest(candidate)}
    receipt['command'][1] = str(runner)
    write(p['receipt'], receipt)
    p['candidate'] = candidate
    monkeypatch.setattr(verdict, 'ROOT', mirror)
    monkeypatch.setattr(verdict, 'RUNNER', mirrored_runner)
    monkeypatch.setattr(verdict, 'git', lambda *args: subprocess.check_output(
        ['git', '-C', str(actual_root), *args], text=True).strip())
    report = write(tmp_path/'report.json', build(p))
    candidate.unlink()
    runner.unlink()
    verdict.verify(report, True)
    (mirror/verdict.CANDIDATE_RECEIPT_V3).write_text('{}')
    with pytest.raises(ValueError, match='missing or edited'):
        verdict.verify(report, True)
