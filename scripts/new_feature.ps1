param(
    [Parameter(Mandatory = $true)]
    [string]$Module,
    [Parameter(Mandatory = $true)]
    [string]$Feature,
    [int]$DefaultWindow = 20
)

$ErrorActionPreference = "Stop"

function Replace-Template {
    param(
        [string]$TemplatePath,
        [string]$OutPath,
        [string]$ModuleName,
        [string]$FeatureName,
        [int]$Window
    )

    $content = Get-Content -Raw -LiteralPath $TemplatePath
    $content = $content.Replace("{{module_name}}", $ModuleName)
    $content = $content.Replace("{{feature_name}}", $FeatureName)
    $content = $content.Replace("{{default_window}}", [string]$Window)
    Set-Content -LiteralPath $OutPath -Value $content -Encoding utf8
}

$root = Split-Path -Parent $PSScriptRoot

$moduleFile = Join-Path $root ("src/quantmaster/features/{0}.py" -f $Module)
if (-not (Test-Path -LiteralPath $moduleFile)) {
    throw "Module file not found: $moduleFile"
}

$testDir = Join-Path $root "tests"
$docDir = Join-Path $root ("docs/features/{0}" -f $Module)
$specDir = Join-Path $root "feature_specs"

New-Item -ItemType Directory -Force -Path $testDir | Out-Null
New-Item -ItemType Directory -Force -Path $docDir | Out-Null
New-Item -ItemType Directory -Force -Path $specDir | Out-Null

$testFile = Join-Path $testDir ("test_{0}.py" -f $Feature)
$docFile = Join-Path $docDir ("{0}.md" -f $Feature)
$specFile = Join-Path $specDir ("{0}.md" -f $Feature)

if (Test-Path -LiteralPath $testFile) { throw "Test file already exists: $testFile" }
if (Test-Path -LiteralPath $docFile) { throw "Doc file already exists: $docFile" }
if (Test-Path -LiteralPath $specFile) { throw "Spec file already exists: $specFile" }

Replace-Template `
    -TemplatePath (Join-Path $root "templates/test_feature.py.tpl") `
    -OutPath $testFile `
    -ModuleName $Module `
    -FeatureName $Feature `
    -Window $DefaultWindow

Replace-Template `
    -TemplatePath (Join-Path $root "templates/doc_feature.md.tpl") `
    -OutPath $docFile `
    -ModuleName $Module `
    -FeatureName $Feature `
    -Window $DefaultWindow

@"
# $Feature

## Hipótese
- TODO

## Fórmula
- TODO

## Entradas
- TODO

## Parâmetros
- window: int = $DefaultWindow

## Saída
- Series nomeada: ${Feature}_${DefaultWindow}

## Regras de borda
- TODO

## Plano de teste
- shape/index
- no-lookahead
- validação de parâmetros
"@ | Set-Content -LiteralPath $specFile -Encoding utf8

Write-Host "Scaffold created:"
Write-Host " - $testFile"
Write-Host " - $docFile"
Write-Host " - $specFile"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) Implement feature in $moduleFile"
Write-Host "2) Export in src/quantmaster/features/__init__.py"
Write-Host "3) Add doc page in mkdocs.yml nav"
Write-Host "4) Run scripts/agent_verify.ps1"

