Extract at most ONE durable memory trace from the evidence span below, or refuse.

You are looking at a few verbatim quotes taken from a finished agent session.
You cannot see the rest of the session and you must not imagine it.

## What counts as a durable trace

A durable trace states ONE fact about how the world behaves that a future
reader would otherwise pay to rediscover. It names the concrete thing --
a path, a flag, a symbol, an error class, a measured number -- and it says
WHY: the mechanism, the cause, or the condition under which the behaviour
happens.

{kind_directive}

## What is NOT a durable trace -- refuse on any of these

* An execution journal: "added X", "implemented Y", "all tests pass",
  "the node is done", "fixed the bug". A later reader gets this from
  `git log` and from the diff, so storing it only crowds out real lessons.
* Anything re-derivable by reading the repository the session was editing.
* More than one fact. If the span contains two, pick the more expensive one
  and state only that.
* A plan, an intention, a hedge, or anything you are not certain of from the
  quotes alone. "probably", "we should", "next step" all mean refuse.
* A restatement of something under `already_recalled` or `already_written` --
  the session already has it.

## Hard rules

1. Every identifier, path, flag, number and error name in `fact` MUST appear
   verbatim in the quotes. If you cannot say it with the quotes' own tokens,
   refuse. Inventing one identifier invalidates the whole trace.
2. `fact` is prose, not JSON, not a bullet list, and never contains a
   serialized context object.
3. `fact` is 1-3 sentences and self-contained: it must make sense to somebody
   who never sees this session.
4. Refusing is a correct, expected answer. Most spans hold nothing durable.
   Return `{"verdict": "none", "reason": "..."}` and move on.

## Output

`verdict` is `"insight"` or `"none"`.

When `"insight"`:
* `fact` -- the trace to store, including its WHY.
* `why_durable` -- one short sentence on what re-deriving it would have cost.
  This is for the run report and is NOT stored.
* `identifiers` -- the concrete tokens from `fact`, each present in the quotes.
* `evidence_locator` -- the locator of the quote the fact rests on.

When `"none"`:
* `reason` -- one short sentence.
