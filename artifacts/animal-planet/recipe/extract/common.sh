# Shared configuration for the animal-planet extraction recipe.
# Sourced by every runnable step. All paths are overridable via environment.
#
# HARD RULES (see recipe/01-extract.md):
# - alt is strictly read-only evidence: never stop/restart/mutate anything
#   there, never write files on alt (remote python runs with the script on
#   stdin only).
# - $AP_STAGING and $AP_TMP hold raw private data. They live OUTSIDE the
#   repo and are never tracked or committed.

: "${AP_STAGING:=/home/sfx/.cache/ap-audit/staging}"
: "${AP_TMP:=/tmp/ap-audit}"
: "${AP_ALT_HOST:=alt}"
: "${AP_ALT_DB_DIR:=/home/user/.local/share/living-memory}"
: "${AP_ALT_TRANSCRIPT_GLOB:=~/.claude/projects/*animal-planet*/*.jsonl}"
: "${AP_LOCAL_DB_DIR:=$HOME/.local/share/living-memory}"
: "${AP_HOLDOUT_SCOPES:=project:octopus,project:online,project:x}"
: "${AP_HOLDOUT_LIMIT:=1500}"

export AP_STAGING AP_TMP AP_ALT_HOST AP_ALT_DB_DIR AP_ALT_TRANSCRIPT_GLOB
export AP_LOCAL_DB_DIR AP_HOLDOUT_SCOPES AP_HOLDOUT_LIMIT

mkdir -p "$AP_STAGING/alt-db" "$AP_STAGING/transcripts" "$AP_STAGING/local-db" \
         "$AP_TMP/alt" "$AP_TMP/local"

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
