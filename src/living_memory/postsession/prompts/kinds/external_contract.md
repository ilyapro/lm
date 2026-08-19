This span is an **external contract**: a probe of a CLI, an API or a service
outside the session's own repository, whose reply documents behaviour. The
durable fact is the contract -- which flag exists, what the endpoint returns,
which status code means what, which argument is rejected.

Refuse if the behaviour belongs to the repository the session was editing:
that is re-derivable by reading its source.
