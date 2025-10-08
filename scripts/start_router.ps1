Param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8787
)

$env:PYTHONPATH = "$PSScriptRoot\..;" + $env:PYTHONPATH
python -m uvicorn src.api.trader_router:app --host $Host --port $Port

