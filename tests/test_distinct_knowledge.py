"""Similarity finds claims without declaring their content or source identical."""
from dataclasses import replace

import pytest

from living_memory.delivery import DeliveryHistory, shape_recall_results
from living_memory.models import Node
from living_memory.retrieval import RecallResult
from living_memory.server import _select_distinct_results, create_mcp_server
from test_lookup import FakeMCP

SCOPE = 'project:distinct-knowledge'
PREFIX = ('Deployment policy. The owner reads the current assignment and saved result, '
          'validates the combined checkout, and preserves operator questions. ' * 3)
PAIRS = [
    ('The operator permits restarting the dashboard.',
     'The operator does not permit restarting the dashboard.'),
    ('Deployment is allowed before approval.', 'Deployment is forbidden before approval.'),
    ('The dashboard can restart during deployment.',
     'The dashboard can restart only after explicit operator approval.'),
]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setenv('LIVING_MEMORY_EMBEDDING_BACKEND', 'hash')
    monkeypatch.setenv('LM_RECALL_MIN_SCORE', 'off')
    monkeypatch.setenv('LM_RECALL_MAP', '0')
    monkeypatch.setenv('LM_DECAY_SWEEP_INTERVAL_SEC', '0')
    monkeypatch.setenv('LM_AUTO_CONSOLIDATE_POLICY', 'off')
    monkeypatch.setenv('LM_DELIVERY_SNIPPET_LADDER', 'off')
    monkeypatch.setenv('LM_DELIVERY_SNIPPET_CHARS', '1600')


@pytest.mark.parametrize('view', ['knowledge', 'full'])
@pytest.mark.parametrize('claims', PAIRS)
def test_ordinary_recall_drain_keeps_distinct_claims(tmp_path, monkeypatch, view, claims):
    # Installed obsolete settings cannot restore semantic mutation on a read.
    monkeypatch.setenv('LM_DRAIN_NEAR_DUP_SUPERSEDES', '1')
    monkeypatch.setenv('LM_RECALL_NEAR_DUP_COSINE', '0.01')
    monkeypatch.setenv('LM_DRAIN_NEAR_DUP_COSINE', '0.01')
    mcp = create_mcp_server(tmp_path / 'memory.sqlite3', mcp_factory=FakeMCP)
    store = mcp.memory_store
    try:
        nodes = [store.append_trace(PREFIX + claim, {'scope': SCOPE}) for claim in claims]
        answer = mcp.tools['memory_recall'](
            'deployment policy dashboard approval', scope=SCOPE, depth=0, max_results=8, view=view,
        )
        entries = {e.get('node', e)['id']: e for e in answer['results']}
        assert set(entries) == {n.id for n in nodes}
        for node in nodes:
            entry = entries[node.id]
            assert entry.get('node', entry)['content'] == node.content
            assert entry['delivery'] == 'full'
            assert entry['superseded'] is False
            assert entry['superseded_by'] == []
            assert store.count_node_chunks(node.id) > 1
        assert store.connection.execute('SELECT count(*) FROM connections').fetchone()[0] == 0
        fetched = mcp.tools['memory_lookup'](node_ids=[n.id for n in nodes], view=view)
        assert {e.get('node', e)['id']: e.get('node', e)['content'] for e in fetched['results']} == {
            n.id: n.content for n in nodes
        }
        event = store.get_recall_event(answer['recall_event_id'])
        assert set(event.result_ids) == set(entries)
    finally:
        store.close()


def result(node_id, content, **kwargs):
    return RecallResult(Node(id=node_id, level='trace', content=content, scope=SCOPE, **kwargs), 1.0)


@pytest.mark.parametrize('distinction', ['content', 'source', 'scope', 'derivation'])
def test_selection_and_delivery_do_not_merge_distinct_identity(distinction):
    first = result('first', 'Deployment is allowed before approval.')
    second = result('second', first.node.content)
    if distinction == 'content':
        second.node.content = 'Deployment is forbidden before approval.'
    elif distinction == 'source':
        first.node.context = {'source': 'operator'}
        second.node.context = {'source': 'reviewer'}
    elif distinction == 'scope':
        second.node.scope = 'project:elsewhere'
    else:
        first.node.level = second.node.level = 'concept'
        first.node.provenance = second.node.provenance = {'strategy': 'token-jaccard'}
        first.node.source_traces = ['source-a']
        second.node.source_traces = ['source-b']
    third = result('third', 'Inventory needs inspection.')
    selected, residual = _select_distinct_results([first, second], [third], 2, DeliveryHistory())
    assert selected == [first, second] and residual == [third]
    shaped = shape_recall_results(selected, view='knowledge', delivery_history=DeliveryHistory(),
                                  snippet_max_chars=1200, context_value_max_chars=160,
                                  session_dedup=False)
    assert [e['delivery'] for e in shaped] == ['full', 'full']
    assert all('duplicate_of' not in e['content_ref'] for e in shaped)


def test_exact_identity_can_release_slot_and_share_delivery():
    first = result('first', 'Inventory requires inspection.')
    twin = result('twin', first.node.content)
    distinct = result('distinct', 'Approval is required before deployment.')
    selected, residual = _select_distinct_results([first, twin], [distinct], 2, DeliveryHistory())
    assert selected == [first, distinct] and residual == [twin]
    shaped = shape_recall_results([first, twin], view='knowledge', delivery_history=DeliveryHistory(),
                                  snippet_max_chars=1200, context_value_max_chars=160,
                                  session_dedup=False)
    assert shaped[1]['delivery'] == 'twin_duplicate'
    assert shaped[1]['content_ref']['duplicate_of'] == 'first'


def test_selection_preserves_cut_without_an_available_replacement():
    first = result('first', 'Inventory requires inspection.')
    twin = result('twin', first.node.content)
    assert _select_distinct_results([first, twin], [], 2, DeliveryHistory()) == ([first, twin], [])


def test_same_content_different_sources_survives_write_and_recall(tmp_path):
    mcp = create_mcp_server(tmp_path / 'memory.sqlite3', mcp_factory=FakeMCP)
    store = mcp.memory_store
    try:
        nodes = [store.append_trace('Deployment requires approval.', {'scope': SCOPE, 'source': source})
                 for source in ('operator', 'reviewer')]
        response = mcp.tools['memory_recall']('deployment approval', scope=SCOPE, max_results=8)
        assert {e['id'] for e in response['results']} == {n.id for n in nodes}
        assert all(e['delivery'] == 'full' for e in response['results'])
        assert all(not store.get_node(n.id).decayed for n in nodes)
    finally:
        store.close()
