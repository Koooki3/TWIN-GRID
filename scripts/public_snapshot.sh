#!/usr/bin/env bash
# Build (and optionally push) the PUBLIC snapshot of this repository: the tree of HEAD minus runs/ruler/ (RULER-prepared
# inputs and prediction files that contain third-party text such as the Paul Graham essays; regenerate them with
# scripts/ruler_prepare.sh), plus a generated PROVENANCE.md (full local commit table, so commit ids cited in the docs,
# e.g. the v1.4 freeze 8e55275, can be traced). Uses a temporary index (read-tree / rm --cached / write-tree /
# commit-tree), so the local branch, its history and the working tree are not changed.
# Author/committer of the snapshot commit: Koooki3 <Koooki3@users.noreply.github.com> (the owner's personal e-mail is
# not published without the owner's explicit choice; the local git config is not changed).
# Checks before any push (exit non-zero on failure; reviewer pre-push list):
#   - the snapshot tree differs from HEAD's tree only by the removal of runs/ruler/ and the addition of PROVENANCE.md;
#   - the sha256 of every frozen file (results/freeze_v14_sha256.txt) recomputed from the snapshot tree matches.
# --push additionally: archives the full local history as a git bundle under ARCHIVE (not pushed), pushes the snapshot
# to main of the public repository (deploy key id_ed25519_twingrid), clones it into a temp dir and re-checks the frozen
# hashes and the CPU-only checks on the clone.
# Usage: public_snapshot.sh [--push]
set -eu
WT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=git@github.com:Koooki3/TWIN-GRID.git
ARCHIVE=/home/kooki3/archives/twin-grid/2026-10-17
PYCHK=${PYCHK:-/home/kooki3/projects/dual-channel-attention/.venv/bin/python}
TMP=$(mktemp -d)
HEAD_ID=$(git -C "$WT" rev-parse HEAD)
# PROVENANCE.md
{
  echo "# Provenance of this public snapshot"
  echo
  echo "This repository is a snapshot of the local development branch at commit \`${HEAD_ID}\`, with \`runs/ruler/\` omitted"
  echo "(RULER-prepared inputs and prediction files containing third-party text; regenerate with \`scripts/ruler_prepare.sh\`)."
  echo "Commit ids cited in the documents refer to the local history below; the full history is archived locally as a git"
  echo "bundle. Frozen files are verified by sha256 (\`results/freeze_v14_sha256.txt\`), independent of commit ids."
  echo
  echo "| commit | date (+0800) | subject |"
  echo "|---|---|---|"
  git -C "$WT" log --format='| %h | %ci | %s |' | sed 's/ +0800 |/ |/'
} > "$TMP/PROVENANCE.md"
export GIT_INDEX_FILE=$TMP/index
git -C "$WT" read-tree HEAD
git -C "$WT" rm -r -q --cached runs/ruler
BLOB=$(git -C "$WT" hash-object -w "$TMP/PROVENANCE.md")
git -C "$WT" update-index --add --cacheinfo 100644 "$BLOB" PROVENANCE.md
TREE=$(git -C "$WT" write-tree)
unset GIT_INDEX_FILE
# check 1: only runs/ruler removed and PROVENANCE.md added
OTHER=$(git -C "$WT" diff-tree -r --name-status HEAD "$TREE" | grep -v -E '^D	runs/ruler/|^A	PROVENANCE.md$' || true)
[ -z "$OTHER" ] || { echo "CHECK FAILED: unexpected tree differences:"; echo "$OTHER"; exit 2; }
echo "check 1 ok: snapshot tree = HEAD tree - runs/ruler/ + PROVENANCE.md ($(git -C "$WT" diff-tree -r HEAD "$TREE" | grep -c 'runs/ruler/') files removed)"
# check 2: frozen sha256 recomputed from the snapshot tree
check_frozen() {   # $1 = git dir or worktree, $2 = tree-ish
  local bad=0
  while read -r h f; do
    rel=${f#"$WT/"}
    got=$(git -C "$1" cat-file -p "$2:$rel" | sha256sum | cut -d' ' -f1)
    [ "$got" = "$h" ] || { echo "  frozen mismatch: $rel"; bad=1; }
  done < "$WT/results/freeze_v14_sha256.txt"
  return $bad
}
check_frozen "$WT" "$TREE" || { echo "CHECK FAILED: frozen hashes"; exit 2; }
echo "check 2 ok: $(wc -l < "$WT/results/freeze_v14_sha256.txt") frozen files match on the snapshot tree"
MSG="TWIN-GRID public snapshot of local ${HEAD_ID:0:7}

runs/ruler/ omitted (third-party text; regenerate with scripts/ruler_prepare.sh). See PROVENANCE.md.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
SSH="ssh -i $HOME/.ssh/id_ed25519_twingrid -o IdentitiesOnly=yes -o BatchMode=yes"
# parent: the current public main (so every update is a fast-forward and never needs a force push); none if empty
REMOTE_MAIN=$(GIT_SSH_COMMAND="$SSH" git -C "$WT" ls-remote "$REMOTE" refs/heads/main | cut -f1)
PARENT=()
if [ -n "$REMOTE_MAIN" ]; then
  GIT_SSH_COMMAND="$SSH" git -C "$WT" fetch -q "$REMOTE" main
  [ "$(git -C "$WT" rev-parse FETCH_HEAD)" = "$REMOTE_MAIN" ] || { echo "fetched main != ls-remote main"; exit 2; }
  PARENT=(-p "$REMOTE_MAIN")
  echo "public main is $REMOTE_MAIN: snapshot will be its child (fast-forward)"
else
  echo "public main is empty: first snapshot, no parent"
fi
COMMIT=$(GIT_AUTHOR_NAME=Koooki3 GIT_AUTHOR_EMAIL=Koooki3@users.noreply.github.com \
         GIT_COMMITTER_NAME=Koooki3 GIT_COMMITTER_EMAIL=Koooki3@users.noreply.github.com \
         git -C "$WT" commit-tree "$TREE" "${PARENT[@]}" -m "$MSG")
echo "snapshot commit $COMMIT (tree $TREE) from local ${HEAD_ID:0:7}"
[ "${1:-}" = "--push" ] || exit 0
mkdir -p "$ARCHIVE"
git -C "$WT" bundle create "$ARCHIVE/twin-grid-local-${HEAD_ID:0:7}.bundle" HEAD
git -C "$WT" bundle verify "$ARCHIVE/twin-grid-local-${HEAD_ID:0:7}.bundle" > /dev/null
echo "archived full local history: $ARCHIVE/twin-grid-local-${HEAD_ID:0:7}.bundle"
GIT_SSH_COMMAND="$SSH" git -C "$WT" push "$REMOTE" "$COMMIT:refs/heads/main"      # never --force: non-fast-forward fails safely
# post-push: clone and re-check
GIT_SSH_COMMAND="$SSH" git clone -q "$REMOTE" "$TMP/clone"
[ "$(git -C "$TMP/clone" rev-parse HEAD)" = "$COMMIT" ] || { echo "POST-PUSH CHECK FAILED: clone HEAD != snapshot"; exit 3; }
check_frozen "$TMP/clone" HEAD || { echo "POST-PUSH CHECK FAILED: frozen hashes on the clone"; exit 3; }
echo "post-push: clone HEAD = snapshot; frozen hashes match on the clone"
bash "$TMP/clone/scripts/check_all.sh" "$PYCHK"
