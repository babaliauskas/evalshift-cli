#!/usr/bin/env bash
# Run every showcase scenario end-to-end (run -> evaluate -> analyze).
#
# Defaults to LIVE: hits the configured models. Pass --offline to use
# the canned fixtures.jsonl shipped with each scenario (no API keys
# required).
#
# Usage:
#   scripts/run_showcase.sh                    # live, all scenarios
#   scripts/run_showcase.sh --offline          # offline, all scenarios
#   scripts/run_showcase.sh --only mix         # one scenario
#   scripts/run_showcase.sh --offline --open   # open the last report

set -euo pipefail

OFFLINE=0
OPEN_LAST=0
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --offline) OFFLINE=1; shift ;;
        --open) OPEN_LAST=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWCASE_DIR="$REPO_ROOT/examples/showcase"

ALL_SCENARIOS=(pass-clean fail-dropped-tool fail-argument-drift mix)
if [[ -n "$ONLY" ]]; then
    SCENARIOS=("$ONLY")
else
    SCENARIOS=("${ALL_SCENARIOS[@]}")
fi

declare -a SUMMARY

run_one () {
    local name="$1"
    local dir="$SHOWCASE_DIR/$name"

    if [[ ! -d "$dir" ]]; then
        echo "skip: $name (no such directory)"
        return
    fi

    echo
    echo "═══════════════════════════════════════════════════════"
    echo "  Scenario: $name  ($([[ $OFFLINE -eq 1 ]] && echo offline || echo live))"
    echo "═══════════════════════════════════════════════════════"

    local run_args=(--yes)
    if [[ $OFFLINE -eq 1 ]]; then
        run_args+=(--offline --fixtures fixtures.jsonl)
    fi

    pushd "$dir" >/dev/null

    evalshift run "${run_args[@]}"
    local run_id
    run_id="$(ls -t .evalshift/runs/ | head -n1)"
    evalshift evaluate "$run_id"
    evalshift analyze "$run_id"

    local report_path=".evalshift/runs/$run_id/report.html"
    SUMMARY+=("$name|$run_id|$dir/$report_path")

    popd >/dev/null
}

for s in "${SCENARIOS[@]}"; do
    run_one "$s"
done

# Build the reports too so opening just works.
for s in "${SCENARIOS[@]}"; do
    dir="$SHOWCASE_DIR/$s"
    [[ -d "$dir" ]] || continue
    run_id="$(ls -t "$dir/.evalshift/runs/" | head -n1)"
    pushd "$dir" >/dev/null
    evalshift report "$run_id" >/dev/null
    popd >/dev/null
done

echo
echo "═══════════════════════════════════════════════════════"
echo "  Summary"
echo "═══════════════════════════════════════════════════════"
printf "%-22s  %-32s  %s\n" "scenario" "run_id" "report"
for row in "${SUMMARY[@]}"; do
    IFS='|' read -r name run_id report <<< "$row"
    printf "%-22s  %-32s  %s\n" "$name" "$run_id" "$report"
done

if [[ $OPEN_LAST -eq 1 && ${#SUMMARY[@]} -gt 0 ]]; then
    last="${SUMMARY[${#SUMMARY[@]}-1]}"
    IFS='|' read -r _ _ last_report <<< "$last"
    if command -v open >/dev/null 2>&1; then
        open "$last_report"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$last_report"
    fi
fi
