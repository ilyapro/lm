# Holdout 6: prospective sfx + alt records for the group-identity candidate

Frozen before any new snapshot is taken or read. It applies only to the
final pinned candidate (production, reader and oracle at the commit named in
`holdout6-result.json`), and only after the candidate's regression replays are
green and production net against `24e9009` is ≤ 0. It is not used to change
the candidate.

Sources: one new read-only online backup of the sfx live store and one of the
alt live store, the alt copy transferred alt→sfx only. Nothing from sfx goes
to alt; private data stays in local scratch, never in git.

Exclusions, decided now: any node whose id or stripped-content SHA-256
appears in any snapshot examined so far (the sfx design snapshot, the alt
holdout-4 snapshot and its items copy, the alt holdout-5 snapshot), and any
record written by this goal's own sessions (context task containing
`procedures-without-case-history`, or task_pattern `c32a8e82d028a68d`).
Report the excluded count and verify zero retained overlap.

Families, classification, queries, reader and verdict: exactly those of
`holdout4.py items/measure/pass/census/report` and `holdout5-protocol.md`
(original `24e9009` on an untouched copy versus the candidate after two
ordinary passes), except that the reader's peer step is inert. All remaining
eligible items enter; no sampling or tuning.

Minimum: 10 eligible instruction/correction items across both families and
both hosts. With fewer, no independent claim is made: the report states the
exact deficit, the candidate stays as pinned, and no replay of an examined
corpus is presented as independent. The regression replays and the gate are
not rerun while waiting for records.
