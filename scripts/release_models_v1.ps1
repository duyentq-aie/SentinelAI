# Creates the (private) GitHub release `models-v1` with the two helmet models. Run it yourself from the repo root, after:
#   winget install GitHub.cli        (once)
#   gh auth login                     (once; the token stays in gh's own store, nobody has to read it)
# The staged files (renamed copies + SHA256SUMS) are in outputs/release/models-v1/ (python tools/verify_models.py outputs/release/models-v1 must say ok).
$ErrorActionPreference = "Stop"
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Write-Host "gh is not installed: winget install GitHub.cli"; exit 1 }
gh auth status 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "gh is not logged in: run  gh auth login  and then this script again"; exit 1 }
python tools/verify_models.py outputs/release/models-v1
if ($LASTEXITCODE -ne 0) { Write-Host "the staged files do not match docs/releases/models-v1.sha256"; exit 1 }
gh release create models-v1 outputs/release/models-v1/helmet_3cls_exp015.pt outputs/release/models-v1/helmet_4cls_exp016.pt outputs/release/models-v1/SHA256SUMS `
  --repo duyentq-aie/SentinelAI --title "models-v1: helmet models (EXP-015 default, EXP-016)" --notes-file docs/releases/models-v1.md
