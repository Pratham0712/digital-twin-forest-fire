<#
deploy_azure.ps1 - one-command deployment of the Forest Fire Digital Twin to Azure App Service.

WHAT IT DOES
  1. Checks Azure CLI + git, signs you in if needed.
  2. Tries each region x plan size (B1, B2, P0V3) until Azure accepts one
     (handles "region not allowed" and "no capacity" errors by moving on).
  3. Creates a Linux Python 3.12 Web App, turns on WebSockets, Always on,
     HTTPS only and sets the Streamlit startup command.
  4. Loads your settings/secrets from azure.env (plain KEY=VALUE lines) into
     the app's environment variables.
  5. Zips the COMMITTED code of your branch (git archive: no venv, no .env)
     and deploys it. Azure installs requirements.txt itself.

USAGE (PowerShell, from the project root, with your changes committed)
  .\deploy_azure.ps1 -AppName forest-fire-twin-bmsce

  Optional:
  .\deploy_azure.ps1 -AppName my-app -Regions southeastasia,uksouth -Skus B2
  .\deploy_azure.ps1 -AppName my-app -SettingsFile my-secrets.env

You need: Azure CLI (winget install -e --id Microsoft.AzureCLI), git, and
an azure.env file next to this script (copy azure.env.example).
Never commit azure.env.
#>
param(
    [Parameter(Mandatory = $true)][string]$AppName,
    [string]$ResourceGroup = "rg-forest-fire",
    [string[]]$Regions = @("centralindia", "southindia", "southeastasia", "eastasia",
                           "uksouth", "westeurope", "northeurope", "eastus", "westus2"),
    [string[]]$Skus = @("B1", "B2", "P0V3"),
    [string]$SettingsFile = "azure.env",
    [string]$Branch = "pp_branch"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "   $m" -ForegroundColor Gray }
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }
function FirstLine($t) {
    $s = ($t | Out-String).Trim() -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1
    if ($s.Length -gt 220) { $s = $s.Substring(0, 220) + "..." }
    return $s
}

# ---------------------------------------------------------------- 1. checks
Step "Checking tools"
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Fail "Azure CLI not found. Run:  winget install -e --id Microsoft.AzureCLI   then open a NEW PowerShell window."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git not found." }

az account show --output none 2>$null
if ($LASTEXITCODE -ne 0) {
    Info "Not signed in to Azure - opening the browser login..."
    az login --output none
    if ($LASTEXITCODE -ne 0) { Fail "az login failed." }
}
$sub = az account show --query "{name:name, id:id}" -o tsv
Info "Subscription: $sub"

$currentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($currentBranch -ne $Branch) {
    Write-Host "   WARNING: you are on branch '$currentBranch', not '$Branch'. Deploying '$currentBranch'." -ForegroundColor Yellow
}
if ((git status --porcelain) -ne $null -and (git status --porcelain).Length -gt 0) {
    Write-Host "   WARNING: you have uncommitted changes. Only COMMITTED code is deployed." -ForegroundColor Yellow
}

# ------------------------------------------------------------ 2. settings file
Step "Reading $SettingsFile"
if (-not (Test-Path $SettingsFile)) {
    Fail "$SettingsFile not found. Copy azure.env.example to azure.env and fill in your values."
}
$settings = [ordered]@{}
foreach ($line in Get-Content $SettingsFile -Encoding UTF8) {
    $t = $line.Trim()
    if (-not $t -or $t.StartsWith("#") -or -not $t.Contains("=")) { continue }
    $i = $t.IndexOf("=")
    $k = $t.Substring(0, $i).Trim()
    $v = $t.Substring($i + 1).Trim()
    if (($v.StartsWith('"') -and $v.EndsWith('"') -and $v.Length -ge 2) -or
        ($v.StartsWith("'") -and $v.EndsWith("'") -and $v.Length -ge 2)) {
        $v = $v.Substring(1, $v.Length - 2)
    }
    if ($k) { $settings[$k] = $v }
}
foreach ($need in @("DATABASE_URL", "DEFAULT_ADMIN_PASSWORD")) {
    if (-not $settings.Contains($need) -or -not $settings[$need]) {
        Write-Host "   WARNING: $need is empty in $SettingsFile." -ForegroundColor Yellow
    }
}
$settings["SCM_DO_BUILD_DURING_DEPLOYMENT"] = "true"
Info "$($settings.Count) settings loaded (values are not printed)."

# ------------------------------------------------------------- 3. provision
$planName = "plan-forest-fire"
$rg = $null; $region = $null; $sku = $null; $ready = $false

foreach ($r in $Regions) {
    foreach ($s in $Skus) {
        $g = "$ResourceGroup-$r"
        Step "Trying plan $s in $r"

        $out = az group create --name $g --location $r --output none 2>&1
        if ($LASTEXITCODE -ne 0) {
            Info "Resource group refused: $(FirstLine $out)"
            break                                   # region not allowed: skip its other sizes
        }
        $out = az appservice plan create --name $planName --resource-group $g `
               --is-linux --sku $s --location $r --output none 2>&1
        if ($LASTEXITCODE -ne 0) {
            Info "Plan refused: $(FirstLine $out)"
            continue
        }

        az webapp show --name $AppName --resource-group $g --output none 2>$null
        if ($LASTEXITCODE -ne 0) {
            $out = az webapp create --name $AppName --resource-group $g `
                   --plan $planName --runtime "PYTHON:3.12" --output none 2>&1
            if ($LASTEXITCODE -ne 0) {
                $msg = FirstLine $out
                if ($msg -match "already exists|not available|already taken|is in use") {
                    Fail "The app name '$AppName' is taken (names are global). Use a different -AppName. ($msg)"
                }
                Info "Web app refused: $msg"
                az group delete --name $g --yes --no-wait --output none 2>$null
                continue
            }
        }
        $rg = $g; $region = $r; $sku = $s; $ready = $true
        break
    }
    if ($ready) { break }
}
if (-not $ready) {
    Fail ("No region/size worked for this subscription. Try other regions with -Regions, " +
          "ask Azure support for App Service quota, or use Streamlit Community Cloud.")
}
Write-Host "`nProvisioned: $AppName on $sku in $region (resource group $rg)" -ForegroundColor Green

# --------------------------------------------------------------- 4. configure
Step "Configuring the web app"
$startup = "python -m streamlit run src/dashboard/app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true"
$out = az webapp config set --name $AppName --resource-group $rg `
       --web-sockets-enabled true --always-on true --startup-file $startup --output none 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Could not set WebSockets / Always on / startup command: $(FirstLine $out)" }
az webapp update --name $AppName --resource-group $rg --https-only true --output none 2>$null

$hostName = (az webapp show --name $AppName --resource-group $rg --query defaultHostName -o tsv).Trim()
$url = "https://$hostName"
if (-not $settings.Contains("APP_URL") -or -not $settings["APP_URL"] -or $settings["APP_URL"] -like "*YOUR-APP*") {
    $settings["APP_URL"] = $url
}

$list = @()
foreach ($k in $settings.Keys) { $list += @{ name = $k; value = [string]$settings[$k]; slotSetting = $false } }
$json = ConvertTo-Json -InputObject $list -Depth 4
$tmp = Join-Path $env:TEMP "azure_appsettings_$([guid]::NewGuid().ToString('N')).json"
[System.IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding $false))
try {
    $out = az webapp config appsettings set --name $AppName --resource-group $rg `
           --settings "@$tmp" --output none 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "Could not set app settings: $(FirstLine $out)" }
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue      # the temp file holds secrets
}
Info "Settings applied."

# ----------------------------------------------------------------- 5. deploy
Step "Packaging committed code ($currentBranch)"
$zip = Join-Path $env:TEMP "forest_fire_app.zip"
Remove-Item $zip -Force -ErrorAction SilentlyContinue
git archive --format=zip --output=$zip HEAD
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $zip)) { Fail "git archive failed." }
Info ("Package: {0:N1} MB" -f ((Get-Item $zip).Length / 1MB))

Step "Deploying (Azure installs requirements.txt - this takes 5-10 minutes)"
$out = az webapp deploy --name $AppName --resource-group $rg --src-path $zip --type zip 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Deploy command reported: $(FirstLine $out)" -ForegroundColor Yellow
    Write-Host "   The build may still be running. Watch it with:" -ForegroundColor Yellow
    Write-Host "   az webapp log tail --name $AppName --resource-group $rg" -ForegroundColor Yellow
}
Remove-Item $zip -Force -ErrorAction SilentlyContinue

# ------------------------------------------------------------------- summary
Write-Host "`n==================== DONE ====================" -ForegroundColor Green
Write-Host "App URL        : $url"
Write-Host "Resource group : $rg   (plan $sku, region $region)"
Write-Host ""
Write-Host "Watch the logs : az webapp log tail --name $AppName --resource-group $rg"
Write-Host "Redeploy later : commit, then run this script again with the same -AppName"
Write-Host "Delete it all  : az group delete --name $rg --yes"
Write-Host ""
Write-Host "Open the URL after 'You can now view your Streamlit app' appears in the log," -ForegroundColor Gray
Write-Host "press Ctrl+Shift+R, then restrict your Google Maps key to $hostName/*" -ForegroundColor Gray
