# Query-anchor reference compatibility review

The two former whole-output assertions compared the current ranker with
`ed20653:src/living_memory/retrieval.py` over the unchanged
`COLD_START_MATRIX`. Its `backup snapshot before mutating database` query
matches the fixture schema's identical trigger. The historical ranker raises
that schema's base score to its trigger score (up to 1.0) and then applies an
unconditional 1.8 trigger multiplier. The frozen applicability candidate
instead lets trigger evidence admit the schema but scales its rank contribution
by content vector evidence, with the graph weight as a small floor. Thus the
old equality claim compared different **non-anchor** trigger semantics. The
backup row was the sole divergence in each original sweep; removing that row
would hide the exact incompatibility.

The revised observable invariant has two parts for every original query,
depth/mode, and scope in the cold-start sweep, and every original query and
depth/mode in the unresembled sweep:

1. With schema trigger collection disabled in **both** arms, current output is
   byte-identical to the independently loaded `ed20653` retrieval implementation.
   Schema nodes remain available through ordinary content and graph channels;
   only the changed trigger evidence is neutralized. This still checks the
   surrounding collection, scoring, graph traversal, and result serialization
   against historical code, including the backup query.
2. With the current trigger behavior enabled, anchor seeding on and off give
   byte-identical output. This checks the actual backup-trigger behavior under
   one consistent rule. The unresembled fixture retains three real unrelated
   anchors; the cold-start fixture retains zero anchors.

The probe also runs an injected mutant of the current service that spuriously
returns a strong anchor seed for a fixture node even at cold start and when
only unrelated anchors exist. For each population, the mutant must differ from
the historical neutral-trigger result. This executable counterexample would
fail if the oracle were insensitive to an anchor-induced ranking change. The
existing real-model jargon retrieval, anchor invisibility, scope isolation,
historical pre-fix demotion fixture, and their assertions are retained.

Executed `npm test -- -q tests/test_retrieval_query_anchors.py`: exit 0,
`20 passed`. This resolves only the query-anchor oracle conflict. It makes no
claim about the separate procedure-retention failures or parent P4; the parent
owns a fresh whole-suite run after integration. Production, v2 receipts,
reports, and private inputs were not changed.
