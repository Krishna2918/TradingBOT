Param(
    [Parameter(Mandatory = $true)] [string] $NewsApiKey,
    [Parameter(Mandatory = $true)] [string] $AlphaVantageKey,
    [Parameter(Mandatory = $true)] [string] $FinnhubKey,
    [Parameter(Mandatory = $false)] [string] $QuestradeRefreshToken,
    [switch] $SetSession,
    [switch] $Persist
)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ErrorActionPreference = 'Stop'

function Mask([string]$s){
    if(-not $s){ return '' }
    $s = $s.Trim()
    if($s.Length -le 8){ return ('*' * $s.Length) }
    return ($s.Substring(0,4) + ('*' * ($s.Length-8)) + $s.Substring($s.Length-4,4))
}

function Show-KeyInfo($name, $val){
    Write-Host ("{0} = [{1}] len={2}" -f $name, (Mask $val), ($val.Trim().Length))
}

function Set-SessionVars(){
    $env:NEWSAPI_KEY       = $NewsApiKey.Trim()
    $env:ALPHAVANTAGE_KEY  = $AlphaVantageKey.Trim()
    $env:FINNHUB_KEY       = $FinnhubKey.Trim()
    if($QuestradeRefreshToken){ $env:QUES_TRADE_REFRESH_TOKEN = $QuestradeRefreshToken.Trim() }
}

function Persist-Vars(){
    cmd /c "setx NEWSAPI_KEY $NewsApiKey" | Out-Null
    cmd /c "setx ALPHAVANTAGE_KEY $AlphaVantageKey" | Out-Null
    cmd /c "setx FINNHUB_KEY $FinnhubKey" | Out-Null
    if($QuestradeRefreshToken){ cmd /c "setx QUES_TRADE_REFRESH_TOKEN $QuestradeRefreshToken" | Out-Null }
}

function Invoke-Get([string]$uri, [int]$timeoutSec = 20){
    try{
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $uri -TimeoutSec $timeoutSec -ErrorAction Stop
        return [pscustomobject]@{ Ok=$true; Status=$resp.StatusCode; Text=$resp.Content }
    } catch {
        $status = if($_.Exception.Response){ $_.Exception.Response.StatusCode.value__ } else { -1 }
        $text = if($_.ErrorDetails){ $_.ErrorDetails.Message } else { '' }
        return [pscustomobject]@{ Ok=$false; Status=$status; Text=$text }
    }
}

function Test-NewsApi(){
    $u = "https://newsapi.org/v2/top-headlines?country=ca&pageSize=2&apiKey=$($NewsApiKey.Trim())"
    $r = Invoke-Get $u 20
    try{ $j = $r.Text | ConvertFrom-Json -ErrorAction Stop } catch { $j = $null }
    $ok = $r.Ok -and $r.Status -eq 200 -and $j -and $j.status -eq 'ok'
    return [pscustomobject]@{ Name='NewsAPI'; Ok=$ok; Status=$r.Status; Raw=$r.Text }
}

function Test-AlphaVantage(){
    $u = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=RY.TO&apikey=$($AlphaVantageKey.Trim())"
    $r = Invoke-Get $u 20
    try{ $j = $r.Text | ConvertFrom-Json -ErrorAction Stop } catch { $j = $null }
    $hasErr = $j -and ($j.'Error Message' -or $j.Note)
    $ok = $r.Ok -and $r.Status -eq 200 -and $j -and $j.'Global Quote' -and -not $hasErr
    return [pscustomobject]@{ Name='AlphaVantage'; Ok=$ok; Status=$r.Status; Raw=$r.Text }
}

function Test-Finnhub(){
    $u = "https://finnhub.io/api/v1/news?category=general&token=$($FinnhubKey.Trim())"
    $r = Invoke-Get $u 20
    $ok = $false
    try{ $j = $r.Text | ConvertFrom-Json -ErrorAction Stop; $ok = ($j -is [System.Array]) -and $r.Status -eq 200 } catch { $ok = $false }
    return [pscustomobject]@{ Name='Finnhub'; Ok=$ok; Status=$r.Status; Raw=$r.Text }
}

function Test-Questrade(){
    if(-not $QuestradeRefreshToken){ return [pscustomobject]@{ Name='Questrade'; Ok=$false; Status=-1; Raw='No refresh token provided' } }
    $url = "https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token=$($QuestradeRefreshToken.Trim())"
    $tries = 4
    $last = $null
    for($i=1; $i -le $tries; $i++){
        $r = Invoke-Get $url 20
        $last = $r
        try{ $j = $r.Text | ConvertFrom-Json -ErrorAction Stop } catch { $j = $null }
        if($r.Ok -and $r.Status -eq 200 -and $j -and $j.access_token -and $j.api_server){
            return [pscustomobject]@{ Name='Questrade'; Ok=$true; Status=200; Raw='access_token issued (masked)' }
        }
        Start-Sleep -Seconds ([math]::Min(10, 2 * $i))
    }
    return [pscustomobject]@{ Name='Questrade'; Ok=$false; Status=$last.Status; Raw=$last.Text }
}

Write-Host 'Verifying provided API keys (masked):' -ForegroundColor Cyan
Show-KeyInfo 'NEWSAPI_KEY' $NewsApiKey
Show-KeyInfo 'ALPHAVANTAGE_KEY' $AlphaVantageKey
Show-KeyInfo 'FINNHUB_KEY' $FinnhubKey
if($QuestradeRefreshToken){ Show-KeyInfo 'QUES_TRADE_REFRESH_TOKEN' $QuestradeRefreshToken }

if($SetSession){ Set-SessionVars; Write-Host 'Session environment variables set.' -ForegroundColor Green }
if($Persist){ Persist-Vars; Write-Host 'Variables persisted with setx (new shells will inherit).' -ForegroundColor Green }

$results = @()
$results += Test-NewsApi
$results += Test-AlphaVantage
$results += Test-Finnhub
if($QuestradeRefreshToken){ $results += Test-Questrade }

Write-Host "`nResults:" -ForegroundColor Cyan
foreach($r in $results){
    $status = if($r.Ok){ 'PASS' } else { 'FAIL' }
    $color = if($r.Ok){ 'Green' } else { 'Red' }
    Write-Host ("[{0}] {1} (HTTP {2})" -f $status, $r.Name, $r.Status) -ForegroundColor $color
    if(-not $r.Ok){
        $snippet = ($r.Raw | Out-String).Trim()
        if($snippet.Length -gt 200){ $snippet = $snippet.Substring(0,200) + '...' }
        Write-Host ("  Snippet: " + $snippet)
    }
}

if(($results | Where-Object { -not $_.Ok }).Count -gt 0){ exit 2 } else { exit 0 }
