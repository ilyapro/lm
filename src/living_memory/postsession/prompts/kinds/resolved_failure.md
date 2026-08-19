This span is a **resolved failure**: a command failed, files changed, the same
command then went green. The durable fact is the *diagnosis*: what the failing
output actually meant and which condition produced it -- not that it was fixed.

Refuse unless the quotes let you name the cause. "The test failed and then
passed" is a journal entry; "pytest picks up `addopts = -q` from pyproject.toml,
so passing `-q` again collapses the report to a single line" is a trace.
