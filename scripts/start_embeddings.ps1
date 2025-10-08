Param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8011
)

$env:PYTHONPATH = "$PSScriptRoot\..;" + $env:PYTHONPATH
python -m uvicorn src.services.embeddings_service:app --host $Host --port $Port

