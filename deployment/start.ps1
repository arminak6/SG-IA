param(
    [switch]$NoBuild,
    [int]$WaitSeconds = 300,
    [switch]$SkipManifest
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $RepositoryRoot "compose.yaml"

if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    $ComposeCommand = "docker-compose"
    $ComposePrefix = @("-f", $ComposeFile)
} elseif (Get-Command docker -ErrorAction SilentlyContinue) {
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose is not available. Install Docker Desktop with Compose."
    }
    $ComposeCommand = "docker"
    $ComposePrefix = @("compose", "-f", $ComposeFile)
} else {
    throw "Docker is not installed or is not available on PATH."
}

$RequiredVolumes = @(
    "sgia_rag_qdrant_data",
    "sgia_rag_data",
    "sgia_rag_docling_model_cache",
    "sgia_wiki_docling_model_cache"
)
foreach ($Volume in $RequiredVolumes) {
    & docker volume inspect $Volume *> $null
    if ($LASTEXITCODE -ne 0) {
        & docker volume create $Volume *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create required Docker volume: $Volume"
        }
    }
}

$Arguments = @($ComposePrefix + @("up", "-d"))
if (-not $NoBuild) {
    $Arguments += "--build"
}
& $ComposeCommand @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "The SG-IA Compose stack failed to start."
}

$InitialValidationArguments = @(
    (Join-Path $PSScriptRoot "validate_deployment.py"),
    "--wait", $WaitSeconds,
    "--skip-manifest"
)
& python @InitialValidationArguments
if ($LASTEXITCODE -ne 0) {
    throw "SG-IA started, but deployment validation failed."
}

if (-not $SkipManifest) {
    & python (Join-Path $PSScriptRoot "bootstrap_knowledge.py")
    if ($LASTEXITCODE -ne 0) {
        throw "SG-IA started, but shared-corpus bootstrap failed."
    }

    & python (Join-Path $PSScriptRoot "validate_deployment.py")
    if ($LASTEXITCODE -ne 0) {
        throw "SG-IA was bootstrapped, but final corpus validation failed."
    }
}
