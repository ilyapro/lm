"""Synthetic counterexamples for the sealed aggregate verdict contract."""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import recall_applicability_verdict as verdict


@pytest.fixture(autouse=True)
def isolate_evidence_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(verdict, 'OLD', tmp_path/'old-report.json')
    monkeypatch.setattr(verdict, 'PREREG', tmp_path/'prereg-v2.md')
    monkeypatch.setattr(verdict, 'PREREG_V3', tmp_path/'prereg-v3.md')
    monkeypatch.setattr(verdict, 'V2_REPORT', tmp_path/'report-v2.json')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


def source_hashes():
    return {name: verdict.digest(ROOT/name) for name in verdict.SOURCE_FILES}


def row(case, arm, *, found=True, irrelevant=3, lookup=False):
    n=len(case['required_facts'])
    hits=list(range(n)) if found else []
    calls=[{'tool':'memory_recall','response_bytes':100,'latency_ms':10.0}]
    if lookup:
        calls.append({'tool':'memory_lookup','response_bytes':30,'latency_ms':3.0})
    necessary=len(calls) if found and n else None
    ranks=list(range(1,irrelevant+1))
    relevant=[irrelevant+1] if n and irrelevant<4 else []
    return {'case_id':case['case_id'],'category':case['category'],'required_fact_count':n,
        'reached_fact_indexes':hits,'sufficient':bool(n and found) or not n and irrelevant==0,
        'necessary_call_count':necessary,'necessary_sequence':[x['tool'] for x in calls] if necessary else None,
        'relevant_ranks':relevant,'irrelevant_ranks':ranks,'irrelevant_top_four':irrelevant,
        'call_count':len(calls),'recall_count':1,'lookup_count':int(lookup),
        'response_bytes':sum(x['response_bytes'] for x in calls),
        'necessary_response_bytes':sum(x['response_bytes'] for x in calls) if necessary else None,
        'warm_latency_ms':sum(x['latency_ms'] for x in calls),'calls':calls}


def packet(tmp_path):
    categories=['concrete','compound','instruction','correction','cross-project','absent-knowledge','large-group']
    dev=[{'case_id':f'd{i}','query':f'development question {i}','scope':None,'depth':1,'max_results':4,
          'category':'concrete','split':'development','topic_group':f'dev{i}',
          'required_facts':[f'development fact {i}'],'acceptable_source_ids':[f'dev-source-{i}']} for i in range(4)]
    hold=[]
    for i in range(14):
        cat=categories[i%7]
        hold.append({'case_id':f'h{i}','query':f'holdout question {i}','scope':None,'depth':1,'max_results':4,
                     'category':cat,'split':'holdout','topic_group':f'group{i}',
                     'required_facts':[] if cat=='absent-knowledge' else [f'holdout fact {i}'],
                     'acceptable_source_ids':[] if cat=='absent-knowledge' else [f'hold-source-{i}']})
    cases=dev+hold
    old=write(tmp_path/'previous.json',{'schema_version':1,'cases':dev+[{
        **c,'query':f'old question {i}','topic_group':f'old-group-{i}',
        'acceptable_source_ids':[] if c['category']=='absent-knowledge' else [f'old-source-{i}']}
        for i,c in enumerate(hold)]})
    gold=write(tmp_path/'gold.json',{'schema_version':1,'cases':cases})
    snapshot=write(tmp_path/'snapshot.sqlite3',{'synthetic':True})
    baseline=write(tmp_path/'baseline.json',{'schema_version':1,'code_head':'b9769d84e3188ee1e646627ebe1b4d454f69d6f9','inputs':{
        'snapshot_sha256':verdict.digest(snapshot),'corpus_sha256':verdict.digest(gold)},
        'outcomes':[{'case_id':c['case_id']} for c in cases]})
    verdict.OLD=write(tmp_path/'old-report.json',{'input_sha256':{'snapshot':verdict.digest(snapshot),'goldset':verdict.digest(old)},'verdict':{'P1':{'pass':False}}})
    seal=write(tmp_path/'seal.json',{'schema_version':1,'baseline_commit':'b9769d84e3188ee1e646627ebe1b4d454f69d6f9',
        'baseline_src_tree':verdict.git('rev-parse','b9769d8:src'),
        'files':{name:{'path':str(path),'sha256':verdict.digest(path)} for name,path in
                 [('snapshot',snapshot),('goldset',gold),('previous_goldset',old),('baseline_outcomes',baseline)]}})
    verdict.PREREG.write_text('\n'.join(v['sha256'] for v in verdict.read(seal)['files'].values()))
    candidate=write(tmp_path/'candidate.json',{'baseline_commit':verdict.read(seal)['baseline_commit'],
        'candidate_commit':verdict.git('rev-parse','HEAD'),'src_tree':verdict.git('rev-parse','HEAD:src'),
        'source_sha256':source_hashes()})
    runner_input=write(tmp_path/'input.json',verdict.runner_cases(cases))
    raw_cases={}
    for c in cases:
        base=row(c,'baseline'); candidate_row=row(c,'candidate',irrelevant=2 if c['case_id'] in ('h0','h1') else 3)
        candidate_row['lost_fact_indexes']=[]
        raw_cases[c['case_id']]={'baseline':base,'candidate':candidate_row}
    raw=write(tmp_path/'raw.json',{'baseline_commit':'b9769d8','snapshot_sha256':verdict.digest(snapshot),
        'arm_order':'alternating by case index',
        'warmup':'create_mcp_server warms LocalEmbeddingModel; fresh process and DB per case and arm',
        'environment':'inherited identically; PYTHONPATH differs only by code root',
        'production_code_delta':verdict.production_delta(),'cases':raw_cases})
    identity='synthetic-once-123'
    consume=write(tmp_path/'consumption.json',{'schema_version':1,'identity':identity,
        'status':'started','seal_sha256':verdict.digest(seal),'runner_input_sha256':verdict.digest(runner_input),
        'candidate_commit':verdict.git('rev-parse','HEAD'),'started_at_utc':'2026-01-01T00:00:00Z'})
    receipt=write(tmp_path/'run-receipt.json',{'schema_version':1,
        'started_at_utc':'2026-01-01T00:00:00Z','ended_at_utc':'2026-01-01T00:01:00Z',
        'seal_sha256':verdict.digest(seal),'runner_sha256':verdict.digest(verdict.RUNNER),
        'runner_input':{'path':str(runner_input),'sha256':verdict.digest(runner_input)},
        'raw':{'path':str(raw),'sha256':verdict.digest(raw)},
        'candidate_receipt':{'path':str(candidate),'sha256':verdict.digest(candidate)},
        'consumption':{'path':str(consume),'sha256':verdict.digest(consume)},
        'consumption_identity':identity,'pre_src_tree':verdict.read(candidate)['src_tree'],
        'post_src_tree':verdict.read(candidate)['src_tree'],
        'pre_source_sha256':source_hashes(),'post_source_sha256':source_hashes(),
        'arms':{arm:{'depth':1,'max_results':4,'schema_trigger':'default'} for arm in ('baseline','candidate')},
        'command':[sys.executable,str(verdict.RUNNER),'--snapshot',str(snapshot),'--cases',str(runner_input),
                   '--out',str(raw)]})
    return {'seal':seal,'raw':raw,'receipt':receipt,'candidate':candidate,'input':runner_input,
            'consume':consume,'gold':gold,'cases':cases}


def v3_packet(tmp_path, monkeypatch):
    p=packet(tmp_path)
    prior=tmp_path/'seal-v2.json'
    prior.write_bytes(p['seal'].read_bytes())
    prior_baseline=tmp_path/'baseline-v2.json'
    prior_baseline.write_bytes((tmp_path/'baseline.json').read_bytes())
    write(verdict.V2_REPORT, {'schema_version':2,'evidence':{'seal':{'sha256':verdict.digest(prior)}},'verdict':{'P1':{'pass':True}}})
    gold=verdict.read(p['gold'])
    for c in gold['cases']:
        if c['split']=='holdout':
            c['query']='successor '+c['query']
            c['topic_group']='successor '+c['topic_group']
            c['acceptable_source_ids']=['successor '+s for s in c['acceptable_source_ids']]
    write(p['gold'],gold); p['cases']=gold['cases']
    snapshot=p['seal'].parent/'snapshot.sqlite3'
    snapshot.unlink()
    with sqlite3.connect(snapshot) as db:
        db.execute('CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT)')
        for c in gold['cases']:
            for source in c['acceptable_source_ids']:
                db.execute('INSERT INTO nodes VALUES (?,?)',(source,' '.join(c['required_facts'])))
    # The v2 seal remains immutable; its original synthetic snapshot is separately bound.
    old_gold=tmp_path/'v2-gold.json'
    prior_data=verdict.read(prior)
    # Keep original and successor snapshot identity equal by using the SQLite bytes
    # for both seals, then refresh the original failed report identity.
    prior_data['files']['snapshot']={'path':str(snapshot),'sha256':verdict.digest(snapshot)}
    prior_data['files']['baseline_outcomes']={'path':str(prior_baseline),'sha256':verdict.digest(prior_baseline)}
    old_gold.write_text(json.dumps({'schema_version':1,'cases':packet_cases_v2(gold)},sort_keys=True)+'\n')
    prior_data['files']['goldset']={'path':str(old_gold),'sha256':verdict.digest(old_gold)}
    old_baseline=verdict.read(prior_baseline)
    old_baseline['inputs']['snapshot_sha256']=verdict.digest(snapshot)
    old_baseline['inputs']['corpus_sha256']=verdict.digest(old_gold)
    write(prior_baseline,old_baseline)
    prior_data['files']['baseline_outcomes']['sha256']=verdict.digest(prior_baseline)
    write(prior,prior_data)
    failed=verdict.read(verdict.OLD)
    failed['input_sha256']['snapshot']=verdict.digest(snapshot)
    write(verdict.OLD,failed)
    write(verdict.V2_REPORT, {'schema_version':2,'evidence':{'seal':{'sha256':verdict.digest(prior)}},'verdict':{'P1':{'pass':True}}})
    seal=verdict.read(p['seal'])
    seal['generation']=3
    seal['prior_seals']=[{'path':str(prior),'sha256':verdict.digest(prior)}]
    seal['files']['snapshot']={'path':str(snapshot),'sha256':verdict.digest(snapshot)}
    seal['files']['goldset']['sha256']=verdict.digest(p['gold'])
    baseline=verdict.read(tmp_path/'baseline.json')
    baseline['inputs']['snapshot_sha256']=verdict.digest(snapshot)
    baseline['inputs']['corpus_sha256']=verdict.digest(p['gold'])
    write(tmp_path/'baseline.json',baseline)
    seal['files']['baseline_outcomes']['sha256']=verdict.digest(tmp_path/'baseline.json')
    write(p['seal'],seal)
    write(p['input'],verdict.runner_cases(gold['cases']))
    raw=verdict.read(p['raw']); raw['snapshot_sha256']=verdict.digest(snapshot)
    write(p['raw'],raw)
    consume=verdict.read(p['consume'])
    consume['seal_sha256']=verdict.digest(p['seal'])
    consume['runner_input_sha256']=verdict.digest(p['input'])
    write(p['consume'],consume)
    receipt=verdict.read(p['receipt']); receipt['seal_sha256']=verdict.digest(p['seal'])
    write(p['receipt'],receipt); refresh(p)
    verdict.PREREG_V3.write_text('\n'.join([*(v['sha256'] for v in seal['files'].values()),
        verdict.digest(verdict.OLD),verdict.digest(verdict.V2_REPORT),verdict.digest(prior)]))
    p['prior']=prior
    return p


def packet_cases_v2(successor_gold):
    cases=copy.deepcopy(successor_gold['cases'])
    for c in cases:
        if c['split']=='holdout':
            c['query']=c['query'].removeprefix('successor ')
            c['topic_group']=c['topic_group'].removeprefix('successor ')
            c['acceptable_source_ids']=[s.removeprefix('successor ') for s in c['acceptable_source_ids']]
    return cases


def refresh(p):
    receipt=verdict.read(p['receipt'])
    for key,path in [('raw',p['raw']),('runner_input',p['input']),('consumption',p['consume'])]:
        receipt[key]['sha256']=verdict.digest(path)
    write(p['receipt'],receipt)
    verdict.PREREG.write_text('\n'.join(v['sha256'] for v in verdict.read(p['seal'])['files'].values()))


def make_report(p, tmp_path):
    report=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    path=write(tmp_path/'report.json',report)
    verdict.verify(path,True)
    return path


def test_build_verify_and_private_content_absent(tmp_path):
    p=packet(tmp_path); report=make_report(p,tmp_path)
    data=verdict.read(report)
    assert data['holdout']['improved_topic_group_count']==2
    assert set(data['results']['holdout']['categories'])==verdict.CATEGORIES
    assert all(c['query'] not in report.read_text() for c in p['cases'])
    assert all(f not in report.read_text() for c in p['cases'] for f in c['required_facts'])
    assert all(s not in report.read_text() for c in p['cases'] for s in c['acceptable_source_ids'])
    assert data['verdict']['P4']['pass'] is False
    cmd=[sys.executable,'-c',
         f"import sys; sys.path.insert(0, {str(ROOT/'scripts')!r}); import recall_applicability_verdict as v; from pathlib import Path; v.OLD=Path({str(verdict.OLD)!r}); v.PREREG=Path({str(verdict.PREREG)!r}); v.main()"]
    output=tmp_path/'cli-report.json'; md=tmp_path/'cli-report.md'
    subprocess.run(cmd+['build','--seal',str(p['seal']),'--raw',str(p['raw']),'--run-receipt',str(p['receipt']),
        '--candidate-receipt',str(p['candidate']),'--output',str(output),'--markdown',str(md)],check=True)
    subprocess.run(cmd+['verify','--report',str(output),'--require-success'],check=True)
    assert md.is_file()


def test_edited_preregistration_is_rejected(tmp_path):
    p=packet(tmp_path)
    report=make_report(p,tmp_path)
    verdict.PREREG.write_text('unrelated preregistration')
    with pytest.raises(ValueError,match='sealed hash absent'):
        verdict.verify(report,True)


def test_frozen_report_survives_removed_execution_worktree(tmp_path,monkeypatch):
    p=packet(tmp_path)
    mirror=tmp_path/'repository'
    (mirror/'artifacts/recall-applicability').mkdir(parents=True)
    (mirror/'scripts').mkdir()
    mirrored_runner=mirror/'scripts/recall_applicability_eval.py'
    mirrored_runner.write_bytes(verdict.RUNNER.read_bytes())
    (mirror/'src').symlink_to(ROOT/'src',target_is_directory=True)
    (mirror/verdict.CANDIDATE_RECEIPT).write_bytes(p['candidate'].read_bytes())
    monkeypatch.setattr(verdict,'ROOT',mirror)
    monkeypatch.setattr(verdict,'RUNNER',mirrored_runner)
    def original_git(*args):
        return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    monkeypatch.setattr(verdict,'git',original_git)
    old_root=tmp_path/'execution-worktree'
    candidate=old_root/verdict.CANDIDATE_RECEIPT
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(p['candidate'].read_bytes())
    runner=old_root/verdict.RUNNER.relative_to(mirror)
    runner.parent.mkdir(parents=True)
    runner.write_bytes(verdict.RUNNER.read_bytes())
    p['candidate']=candidate
    consume=verdict.read(p['consume'])
    consume['candidate_commit']=verdict.read(candidate)['candidate_commit']
    write(p['consume'],consume)
    receipt=verdict.read(p['receipt'])
    receipt['candidate_receipt']={'path':str(candidate),'sha256':verdict.digest(candidate)}
    receipt['command'][1]=str(runner)
    write(p['receipt'],receipt); refresh(p)
    report=make_report(p,tmp_path)
    frozen={path:path.read_bytes() for path in (report,p['receipt'],p['consume'],p['raw'],p['seal'])}
    candidate.unlink(); runner.unlink()
    verdict.verify(report,True)
    assert all(path.read_bytes()==before for path,before in frozen.items())

    entry=receipt['candidate_receipt']
    with pytest.raises(ValueError,match='missing or edited'):
        verdict.bound({**entry,'sha256':'0'*64},'candidate receipt',verdict.CANDIDATE_RECEIPT)
    with pytest.raises(ValueError,match='repository-relative'):
        verdict.bound({**entry,'path':str(old_root/'unrelated.json')},'candidate receipt',verdict.CANDIDATE_RECEIPT)
    candidate.write_text('edited')
    with pytest.raises(ValueError,match='missing or edited'): verdict.verify(report,True)
    candidate.unlink()
    runner.write_text('edited')
    with pytest.raises(ValueError,match='command mismatch'): verdict.verify(report,True)
    runner.unlink()
    changed=verdict.read(p['receipt']); changed['runner_sha256']='0'*64
    write(p['receipt'],changed)
    with pytest.raises(ValueError,match='runner hash'):
        verdict.build_data(p['seal'],p['raw'],p['receipt'],mirror/verdict.CANDIDATE_RECEIPT)
    write(p['receipt'],receipt); refresh(p)
    changed=verdict.read(p['receipt']); changed['command'][1]=str(tmp_path/'unrelated/scripts'/runner.name)
    write(p['receipt'],changed)
    with pytest.raises(ValueError,match='command mismatch'):
        verdict.build_data(p['seal'],p['raw'],p['receipt'],mirror/verdict.CANDIDATE_RECEIPT)


def test_one_group_and_duplicate_group_fail(tmp_path):
    p=packet(tmp_path); raw=verdict.read(p['raw'])
    raw['cases']['h1']['candidate']['irrelevant_ranks']=[1,2,3]
    raw['cases']['h1']['candidate']['irrelevant_top_four']=3
    raw['cases']['h1']['candidate']['relevant_ranks']=[4]
    write(p['raw'],raw); refresh(p)
    failed=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    assert failed['holdout']['improved_topic_group_count']==1 and not failed['verdict']['P1']['pass']
    path=write(tmp_path/'failed.json',failed)
    verdict.verify(path)
    with pytest.raises(ValueError,match='P1 threshold'): verdict.verify(path,True)
    cmd=[sys.executable,'-c', f"import sys; sys.path.insert(0, {str(ROOT/'scripts')!r}); import recall_applicability_verdict as v; from pathlib import Path; v.OLD=Path({str(verdict.OLD)!r}); v.PREREG=Path({str(verdict.PREREG)!r}); v.main()"]
    assert subprocess.run(cmd+['verify','--report',str(path),'--require-success'],capture_output=True).returncode != 0
    forged=verdict.read(path); forged['verdict']['P1']['pass']=True; write(path,forged)
    with pytest.raises(ValueError,match='forged'): verdict.verify(path)
    gold=verdict.read(p['gold']); gold['cases'][5]['topic_group']=gold['cases'][4]['topic_group']
    write(p['gold'],gold)
    # A duplicate group is counted once even when both cases improve.
    seal=verdict.read(p['seal']); seal['files']['goldset']['sha256']=verdict.digest(p['gold']); write(p['seal'],seal)
    inp=verdict.runner_cases(gold['cases']); write(p['input'],inp)
    base=verdict.read(tmp_path/'baseline.json'); base['inputs']['corpus_sha256']=verdict.digest(p['gold']); write(tmp_path/'baseline.json',base)
    seal['files']['baseline_outcomes']['sha256']=verdict.digest(tmp_path/'baseline.json'); write(p['seal'],seal)
    consume=verdict.read(p['consume']); consume['seal_sha256']=verdict.digest(p['seal']); consume['runner_input_sha256']=verdict.digest(p['input']); write(p['consume'],consume)
    receipt=verdict.read(p['receipt']); receipt['seal_sha256']=verdict.digest(p['seal']); write(p['receipt'],receipt); refresh(p)
    raw=verdict.read(p['raw']); raw['cases']['h1']['candidate']['irrelevant_ranks']=[1,2]; raw['cases']['h1']['candidate']['irrelevant_top_four']=2
    raw['cases']['h1']['candidate']['relevant_ranks']=[3]
    write(p['raw'],raw); refresh(p)
    assert verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])['holdout']['improved_topic_group_count']==1


@pytest.mark.parametrize('category', ['instruction','correction'])
def test_protected_loss_and_baseline_miss(tmp_path,category):
    p=packet(tmp_path); raw=verdict.read(p['raw'])
    i=next(c['case_id'] for c in p['cases'] if c['split']=='holdout' and c['category']==category)
    raw['cases'][i]['candidate']['reached_fact_indexes']=[]
    raw['cases'][i]['candidate']['sufficient']=False
    raw['cases'][i]['candidate']['necessary_call_count']=None
    raw['cases'][i]['candidate']['necessary_sequence']=None
    raw['cases'][i]['candidate']['necessary_response_bytes']=None
    raw['cases'][i]['candidate']['lost_fact_indexes']=[0]
    # An unrelated baseline miss is recorded separately.
    miss='h7'
    raw['cases'][miss]['baseline']['reached_fact_indexes']=[]
    raw['cases'][miss]['baseline']['sufficient']=False
    raw['cases'][miss]['baseline']['necessary_call_count']=None
    raw['cases'][miss]['baseline']['necessary_sequence']=None
    raw['cases'][miss]['baseline']['necessary_response_bytes']=None
    write(p['raw'],raw); refresh(p)
    data=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    assert not data['verdict']['P1']['pass']
    assert data['holdout']['new_fact_losses'][category]==1
    assert data['holdout']['baseline_missed_facts']['concrete']==1
    assert data['results']['holdout']['categories'][category]['candidate']['exhausted_calls_unreached_cases']>0


def test_necessary_lookup_recovers_fact(tmp_path):
    p=packet(tmp_path); raw=verdict.read(p['raw'])
    case=p['cases'][4]
    for arm in ('baseline','candidate'):
        raw['cases']['h0'][arm]=row(case,arm,lookup=True,irrelevant=3 if arm=='baseline' else 2)
    raw['cases']['h0']['candidate']['lost_fact_indexes']=[]
    write(p['raw'],raw); refresh(p)
    report=make_report(p,tmp_path)
    assert verdict.read(report)['results']['holdout']['categories']['concrete']['candidate']['necessary_calls_reached_cases']>=2


def test_stale_and_forged_evidence(tmp_path):
    p=packet(tmp_path); path=make_report(p,tmp_path)
    forged=verdict.read(path); forged['verdict']['P1']['pass']=False
    write(path,forged)
    with pytest.raises(ValueError,match='forged'): verdict.verify(path)
    path=make_report(p,tmp_path)
    receipt=verdict.read(p['receipt']); receipt['runner_sha256']='0'*64; write(p['receipt'],receipt)
    with pytest.raises(ValueError,match='runner hash'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    receipt['runner_sha256']=verdict.digest(verdict.RUNNER); receipt['post_source_sha256'][verdict.SOURCE_FILES[0]]='0'*64
    write(p['receipt'],receipt)
    with pytest.raises(ValueError,match='source'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    receipt['post_source_sha256']=source_hashes(); write(p['receipt'],receipt)
    inp=verdict.read(p['input']); inp['cases'][0]['query']='edited'
    write(p['input'],inp)
    consume=verdict.read(p['consume']); consume['runner_input_sha256']=verdict.digest(p['input']); write(p['consume'],consume); refresh(p)
    with pytest.raises(ValueError,match='shape-only'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])


def test_missing_categories_and_forbidden_mode(tmp_path):
    p=packet(tmp_path)
    raw=verdict.read(p['raw']); raw['cases'].pop('h5'); write(p['raw'],raw); refresh(p)
    with pytest.raises(ValueError,match='population'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    raw=verdict.read(p['raw']); raw['cases']['h5']={'baseline':row(p['cases'][10],'baseline'),
        'candidate':{**row(p['cases'][10],'candidate'),'lost_fact_indexes':[]}}; write(p['raw'],raw); refresh(p)
    with pytest.raises(ValueError): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    p=packet(tmp_path); receipt=verdict.read(p['receipt']); receipt['arms']['candidate']['schema_trigger']='name'
    write(p['receipt'],receipt)
    with pytest.raises(ValueError,match='forbidden name'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])


@pytest.mark.parametrize('removed_category',['absent-knowledge','large-group'])
def test_required_holdout_category_cannot_disappear(tmp_path,removed_category):
    p=packet(tmp_path)
    gold=verdict.read(p['gold']); raw=verdict.read(p['raw'])
    for case in gold['cases']:
        if case['split']=='holdout' and case['category']==removed_category:
            case['category']='concrete'
            for arm in ('baseline','candidate'):
                raw['cases'][case['case_id']][arm]['category']='concrete'
    write(p['gold'],gold); write(p['raw'],raw)
    write(p['input'],verdict.runner_cases(gold['cases']))
    baseline=verdict.read(tmp_path/'baseline.json')
    baseline['inputs']['corpus_sha256']=verdict.digest(p['gold'])
    write(tmp_path/'baseline.json',baseline)
    seal=verdict.read(p['seal'])
    seal['files']['goldset']['sha256']=verdict.digest(p['gold'])
    seal['files']['baseline_outcomes']['sha256']=verdict.digest(tmp_path/'baseline.json')
    write(p['seal'],seal)
    consume=verdict.read(p['consume'])
    consume['seal_sha256']=verdict.digest(p['seal'])
    consume['runner_input_sha256']=verdict.digest(p['input'])
    write(p['consume'],consume)
    receipt=verdict.read(p['receipt']); receipt['seal_sha256']=verdict.digest(p['seal'])
    write(p['receipt'],receipt); refresh(p)
    with pytest.raises(ValueError,match='holdout coverage'):
        verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])


def test_v3_bound_success_and_preflight(tmp_path,monkeypatch):
    p=v3_packet(tmp_path,monkeypatch)
    assert verdict.validate_seal(p['seal'])[0]['generation']==3
    monkeypatch.setattr(sys,'argv',['verdict','validate-seal','--seal',str(p['seal'])])
    verdict.main()
    report=make_report(p,tmp_path)
    data=verdict.read(report)
    assert data['schema_version']==3
    assert data['holdout']['baseline_reached_protected_facts']=={
        'instruction':2,'correction':2,'cross-project':2}
    assert data['development']['new_fact_losses']=={'concrete':0}


def test_v2_shape_and_nonvacuous_success(tmp_path):
    p=packet(tmp_path)
    report=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    assert report['schema_version']==2
    assert 'development' not in report and 'baseline_reached_protected_facts' not in report['holdout']
    assert 'historical_v2_report_sha256' not in report['evidence']
    raw=verdict.read(p['raw'])
    for c in p['cases']:
        if c['split']=='holdout' and c['category']=='instruction':
            for arm in ('baseline','candidate'):
                raw['cases'][c['case_id']][arm]=row(c,arm,found=False)
            raw['cases'][c['case_id']]['candidate']['lost_fact_indexes']=[]
    write(p['raw'],raw); refresh(p)
    failed=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    assert failed['holdout']['new_fact_losses']['instruction']==0
    assert failed['verdict']['P1']['pass'] is False
    path=write(tmp_path/'report.json',failed)
    verdict.verify(path)
    with pytest.raises(ValueError,match='threshold'): verdict.verify(path,True)


def test_v2_report_rejected_after_current_source_revision(tmp_path,monkeypatch):
    p=packet(tmp_path)
    path=make_report(p,tmp_path)
    original=verdict.git
    def moved_source(*args):
        if args==('rev-parse','HEAD:src'):
            return '0'*40
        return original(*args)
    monkeypatch.setattr(verdict,'git',moved_source)
    with pytest.raises(ValueError,match='current source tree'): verdict.verify(path,True)


@pytest.mark.parametrize('category',['instruction','correction','cross-project'])
def test_v3_requires_reached_protected_control(tmp_path,monkeypatch,category):
    p=v3_packet(tmp_path,monkeypatch)
    raw=verdict.read(p['raw'])
    for c in p['cases']:
        if c['split']=='holdout' and c['category']==category:
            for arm in ('baseline','candidate'):
                raw['cases'][c['case_id']][arm]=row(c,arm,found=False,irrelevant=3)
            raw['cases'][c['case_id']]['candidate']['lost_fact_indexes']=[]
    write(p['raw'],raw); refresh(p)
    report=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    assert report['holdout']['baseline_reached_protected_facts'][category]==0
    assert report['holdout']['new_fact_losses'][category]==0
    assert report['verdict']['P1']['pass'] is False
    path=write(tmp_path/'report.json',report)
    verdict.verify(path)
    with pytest.raises(ValueError,match='P1 threshold'): verdict.verify(path,True)


@pytest.mark.parametrize('case_id',['d0','h2','h3','h4'])
def test_v3_rejects_new_fact_loss(tmp_path,monkeypatch,case_id):
    p=v3_packet(tmp_path,monkeypatch)
    raw=verdict.read(p['raw']); c=next(c for c in p['cases'] if c['case_id']==case_id)
    raw['cases'][case_id]['candidate']=row(c,'candidate',found=False)
    raw['cases'][case_id]['candidate']['lost_fact_indexes']=[0]
    write(p['raw'],raw); refresh(p)
    report=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    losses=report['development'] if case_id.startswith('d') else report['holdout']
    assert losses['new_fact_losses'][c['category']]==1
    assert report['verdict']['P1']['pass'] is False


@pytest.mark.parametrize('population',['original','v2'])
@pytest.mark.parametrize('field',['query','topic_group','acceptable_source_ids'])
def test_v3_rejects_prior_overlap(tmp_path,monkeypatch,population,field):
    p=v3_packet(tmp_path,monkeypatch)
    gold=verdict.read(p['gold']); prior=verdict.read(p['prior'])
    source=verdict.read(tmp_path/'previous.json' if population=='original' else Path(prior['files']['goldset']['path']))
    current=next(c for c in gold['cases'] if c['split']=='holdout' and c['category']=='concrete')
    older=next(c for c in source['cases'] if c['split']=='holdout' and c['category']=='concrete')
    current[field]=copy.deepcopy(older[field])
    write(p['gold'],gold)
    seal=verdict.read(p['seal']); seal['files']['goldset']['sha256']=verdict.digest(p['gold'])
    baseline=verdict.read(tmp_path/'baseline.json')
    baseline['inputs']['corpus_sha256']=verdict.digest(p['gold'])
    write(tmp_path/'baseline.json',baseline)
    seal['files']['baseline_outcomes']['sha256']=verdict.digest(tmp_path/'baseline.json')
    write(p['seal'],seal)
    verdict.PREREG_V3.write_text('\n'.join([*(v['sha256'] for v in seal['files'].values()),
        verdict.digest(verdict.OLD),verdict.digest(verdict.V2_REPORT),verdict.digest(p['prior'])]))
    with pytest.raises(ValueError,match='overlap'): verdict.validate_seal(p['seal'])


def test_v3_edited_prior_seal_and_wrong_generation(tmp_path,monkeypatch):
    p=v3_packet(tmp_path,monkeypatch)
    prior=verdict.read(p['prior']); prior['baseline_commit']='0'*40; write(p['prior'],prior)
    with pytest.raises(ValueError,match='edited'): verdict.validate_seal(p['seal'])
    p=v3_packet(tmp_path,monkeypatch)
    seal=verdict.read(p['seal']); seal['generation']=4; write(p['seal'],seal)
    with pytest.raises(ValueError,match='generation'): verdict.validate_seal(p['seal'])


def test_v3_rejects_wrong_candidate_path_and_source_runner_hashes(tmp_path,monkeypatch):
    p=v3_packet(tmp_path,monkeypatch)
    receipt=verdict.read(p['receipt'])
    receipt['candidate_receipt']['path']=str(tmp_path/'wrong.json')
    write(p['receipt'],receipt)
    with pytest.raises(ValueError,match='candidate receipt'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    p=v3_packet(tmp_path,monkeypatch)
    receipt=verdict.read(p['receipt']); receipt['runner_sha256']='0'*64
    write(p['receipt'],receipt)
    with pytest.raises(ValueError,match='runner hash'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    p=v3_packet(tmp_path,monkeypatch)
    candidate=verdict.read(p['candidate']); candidate['source_sha256'][verdict.SOURCE_FILES[0]]='0'*64
    write(p['candidate'],candidate)
    receipt=verdict.read(p['receipt']); receipt['candidate_receipt']['sha256']=verdict.digest(p['candidate'])
    write(p['receipt'],receipt)
    with pytest.raises(ValueError,match='current source hash'): verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])


def test_v3_insufficient_groups_forgery_and_lookup_recovery(tmp_path,monkeypatch):
    p=v3_packet(tmp_path,monkeypatch)
    raw=verdict.read(p['raw'])
    raw['cases']['h1']['candidate']=row(p['cases'][5],'candidate',irrelevant=3)
    raw['cases']['h1']['candidate']['lost_fact_indexes']=[]
    write(p['raw'],raw); refresh(p)
    report=verdict.build_data(p['seal'],p['raw'],p['receipt'],p['candidate'])
    assert report['holdout']['improved_topic_group_count']==1
    path=write(tmp_path/'report.json',report)
    verdict.verify(path)
    with pytest.raises(ValueError,match='threshold'): verdict.verify(path,True)
    report['verdict']['P1']['pass']=True; write(path,report)
    with pytest.raises(ValueError,match='forged'): verdict.verify(path)
    p=v3_packet(tmp_path,monkeypatch)
    raw=verdict.read(p['raw']); case=p['cases'][4]
    for arm in ('baseline','candidate'):
        raw['cases']['h0'][arm]=row(case,arm,lookup=True,irrelevant=3 if arm=='baseline' else 2)
    raw['cases']['h0']['candidate']['lost_fact_indexes']=[]
    write(p['raw'],raw); refresh(p)
    report=make_report(p,tmp_path)
    assert verdict.read(report)['results']['holdout']['categories']['concrete']['candidate']['necessary_calls_reached_cases']>=2


@pytest.mark.parametrize('category',['absent-knowledge','large-group'])
def test_v3_missing_category_rejected_by_preflight(tmp_path,monkeypatch,category):
    p=v3_packet(tmp_path,monkeypatch)
    gold=verdict.read(p['gold'])
    for case in gold['cases']:
        if case['split']=='holdout' and case['category']==category:
            case['category']='concrete'
    write(p['gold'],gold)
    seal=verdict.read(p['seal']); seal['files']['goldset']['sha256']=verdict.digest(p['gold'])
    baseline=verdict.read(tmp_path/'baseline.json')
    baseline['inputs']['corpus_sha256']=verdict.digest(p['gold'])
    write(tmp_path/'baseline.json',baseline)
    seal['files']['baseline_outcomes']['sha256']=verdict.digest(tmp_path/'baseline.json')
    write(p['seal'],seal)
    verdict.PREREG_V3.write_text('\n'.join([*(v['sha256'] for v in seal['files'].values()),
        verdict.digest(verdict.OLD),verdict.digest(verdict.V2_REPORT),verdict.digest(p['prior'])]))
    with pytest.raises(ValueError,match='category'): verdict.validate_seal(p['seal'])
