#!/usr/bin/env bash
# deploy_azure.sh - one-command deployment of the Forest Fire Digital Twin to Azure App Service.
#
# Run it in Azure Cloud Shell (Bash) or any bash with Azure CLI, git, python3 and zip.
#
#   bash deploy_azure.sh APP_NAME [SETTINGS_FILE]
#   bash deploy_azure.sh forest-fire-twin-bmsce
#
# What it does
#   1. Gets your code: uses the current folder if it is the project, otherwise
#      clones BRANCH from REPO_URL (default: pp_branch on GitHub).
#   2. Tries each region x plan size until Azure accepts one (skips "region not
#      allowed" and "no capacity" errors).
#   3. Creates a Linux Python 3.12 Web App; turns on WebSockets, Always on,
#      HTTPS only; sets the Streamlit startup command.
#   4. Loads azure.env (plain KEY=VALUE lines) into the app's environment variables.
#   5. Zips the COMMITTED code (git archive) and deploys it. Azure installs
#      requirements.txt itself.
#
# Optional environment overrides (space-separated lists):
#   REGIONS="southeastasia uksouth"   SKUS="B2"   BRANCH=pp_branch   RG_BASE=rg-forest-fire

set -u

APP_NAME="${1:-}"
SETTINGS_FILE="${2:-azure.env}"
RG_BASE="${RG_BASE:-rg-forest-fire}"
BRANCH="${BRANCH:-pp_branch}"
REPO_URL="${REPO_URL:-https://github.com/Pratham0712/digital-twin-forest-fire.git}"
# shellcheck disable=SC2206
REGIONS=(${REGIONS:-centralindia southindia southeastasia eastasia uksouth westeurope northeurope eastus westus2})
# shellcheck disable=SC2206
SKUS=(${SKUS:-B1 B2 P0V3})
PLAN_NAME="plan-forest-fire"
STARTUP="python -m streamlit run src/dashboard/app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true"

step() { printf '\n\033[36m== %s\033[0m\n' "$1"; }
info() { printf '   %s\n' "$1"; }
warn() { printf '\033[33m   WARNING: %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mERROR: %s\033[0m\n' "$1"; exit 1; }
last() { printf '%s' "$1" | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c1-220; }

[ -n "$APP_NAME" ] || fail "Usage: bash deploy_azure.sh APP_NAME [SETTINGS_FILE]   (APP_NAME must be globally unique)"

# ------------------------------------------------------------------ checks
step "Checking tools"
for t in az git python3 zip; do
  command -v "$t" >/dev/null 2>&1 || fail "'$t' not found."
done
if ! az account show --output none 2>/dev/null; then
  info "Not signed in - starting device login..."
  az login --use-device-code --output none || fail "az login failed."
fi
info "Subscription: $(az account show --query name -o tsv)"

# ------------------------------------------------------- settings file path
[ -f "$SETTINGS_FILE" ] || fail "$SETTINGS_FILE not found. Create it (see azure.env.example) in this folder."
SETTINGS_FILE="$(cd "$(dirname "$SETTINGS_FILE")" && pwd)/$(basename "$SETTINGS_FILE")"

# ------------------------------------------------------------------ code
step "Getting the code"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ -f src/dashboard/app.py ]; then
  info "Using the project in the current folder ($(git rev-parse --abbrev-ref HEAD))."
  if [ -n "$(git status --porcelain)" ]; then warn "Uncommitted changes exist - only COMMITTED code is deployed."; fi
else
  CODE_DIR="$HOME/forest-fire-app"
  rm -rf "$CODE_DIR"
  info "Cloning $BRANCH from $REPO_URL ..."
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$CODE_DIR" >/dev/null 2>&1 \
    || fail "Clone failed. Is the repo public, and is branch '$BRANCH' pushed to GitHub?"
  cd "$CODE_DIR" || fail "Cannot enter $CODE_DIR"
fi
[ -f src/dashboard/app.py ] || fail "src/dashboard/app.py not found - wrong folder?"
info "Latest commit: $(git log -1 --pretty='%h %s' | cut -c1-90)"

# --------------------------------------------------------------- provision
READY=0; RG=""; REGION=""; SKU=""
for r in "${REGIONS[@]}"; do
  for s in "${SKUS[@]}"; do
    g="$RG_BASE-$r"
    step "Trying plan $s in $r"

    if ! out=$(az group create --name "$g" --location "$r" --output none 2>&1); then
      info "Resource group refused: $(last "$out")"
      break                                     # region not allowed: skip its other sizes
    fi
    if ! out=$(az appservice plan create --name "$PLAN_NAME" --resource-group "$g" \
               --is-linux --sku "$s" --location "$r" --output none 2>&1); then
      info "Plan refused: $(last "$out")"
      continue
    fi
    if ! az webapp show --name "$APP_NAME" --resource-group "$g" --output none 2>/dev/null; then
      if ! out=$(az webapp create --name "$APP_NAME" --resource-group "$g" \
                 --plan "$PLAN_NAME" --runtime "PYTHON:3.12" --output none 2>&1); then
        msg="$(last "$out")"
        if printf '%s' "$msg" | grep -qiE "already exists|not available|already taken|is in use"; then
          fail "The app name '$APP_NAME' is taken (names are global). Use another name. ($msg)"
        fi
        info "Web app refused: $msg"
        az group delete --name "$g" --yes --no-wait --output none 2>/dev/null
        continue
      fi
    fi
    RG="$g"; REGION="$r"; SKU="$s"; READY=1
    break
  done
  [ "$READY" -eq 1 ] && break
done
[ "$READY" -eq 1 ] || fail "No region/size worked for this subscription. Try REGIONS=\"...\" SKUS=\"...\", ask Azure support for quota, or keep Streamlit Community Cloud."
printf '\n\033[32mProvisioned: %s on %s in %s (resource group %s)\033[0m\n' "$APP_NAME" "$SKU" "$REGION" "$RG"

# --------------------------------------------------------------- configure
step "Configuring the web app"
out=$(az webapp config set --name "$APP_NAME" --resource-group "$RG" \
      --web-sockets-enabled true --always-on true --startup-file "$STARTUP" --output none 2>&1) \
  || fail "Could not set WebSockets / Always on / startup command: $(last "$out")"
az webapp update --name "$APP_NAME" --resource-group "$RG" --https-only true --output none 2>/dev/null

HOST="$(az webapp show --name "$APP_NAME" --resource-group "$RG" --query defaultHostName -o tsv | tr -d '\r')"
URL="https://$HOST"

TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT
python3 - "$SETTINGS_FILE" "$TMP_JSON" "$URL" <<'PY'
import json, sys
src, dst, url = sys.argv[1:4]
d = {}
for line in open(src, encoding="utf-8-sig"):
    t = line.strip()
    if not t or t.startswith("#") or "=" not in t:
        continue
    k, v = t.split("=", 1)
    k, v = k.strip(), v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    if k:
        d[k] = v
d["SCM_DO_BUILD_DURING_DEPLOYMENT"] = "true"
if not d.get("APP_URL") or "YOUR-APP" in d["APP_URL"]:
    d["APP_URL"] = url
for need in ("DATABASE_URL", "DEFAULT_ADMIN_PASSWORD"):
    if not d.get(need):
        print(f"   WARNING: {need} is empty in the settings file.")
json.dump([{"name": k, "value": v, "slotSetting": False} for k, v in d.items()], open(dst, "w"))
print(f"   {len(d)} settings loaded (values are not printed).")
PY
out=$(az webapp config appsettings set --name "$APP_NAME" --resource-group "$RG" \
      --settings "@$TMP_JSON" --output none 2>&1) \
  || fail "Could not set app settings: $(last "$out")"
rm -f "$TMP_JSON"
info "Settings applied."

# ------------------------------------------------------------------ deploy
step "Packaging committed code"
ZIP="$(mktemp -u).zip"
git archive --format=zip --output="$ZIP" HEAD || fail "git archive failed."
info "Package: $(du -h "$ZIP" | cut -f1)"

step "Deploying (Azure installs requirements.txt - 5 to 10 minutes)"
if ! out=$(az webapp deploy --name "$APP_NAME" --resource-group "$RG" --src-path "$ZIP" --type zip 2>&1); then
  warn "Deploy command reported: $(last "$out")"
  warn "The build may still be running. Watch it with: az webapp log tail --name $APP_NAME --resource-group $RG"
fi
rm -f "$ZIP"

# ----------------------------------------------------------------- summary
printf '\n\033[32m==================== DONE ====================\033[0m\n'
echo "App URL        : $URL"
echo "Resource group : $RG   (plan $SKU, region $REGION)"
echo
echo "Watch the logs : az webapp log tail --name $APP_NAME --resource-group $RG"
echo "Redeploy later : push to GitHub, then run this script again with the same APP_NAME"
echo "Delete it all  : az group delete --name $RG --yes"
echo
echo "Open the URL once the log shows 'You can now view your Streamlit app', then press Ctrl+Shift+R."
echo "Then restrict your Google Maps key to ${HOST}/*"
