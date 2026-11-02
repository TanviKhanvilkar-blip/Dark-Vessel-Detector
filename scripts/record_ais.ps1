# Keeps the AIS recorder running and restarts it if Python crashes. Stop with Ctrl+C.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File scripts\record_ais.ps1
param([string]$Db = "live.sqlite")

if (-not $env:AISSTREAM_KEY) {
    # Asked for here so the key is never typed on a command line or saved in shell history.
    $secure = Read-Host "aisstream.io API key" -AsSecureString
    $env:AISSTREAM_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

while ($true) {
    python -m dvd record-ais --db $Db
    if ($LASTEXITCODE -eq 0) { break }                                  # stopped on purpose
    if ($LASTEXITCODE -eq 1) { Write-Host "API key rejected; not restarting."; break }
    Write-Host "Recorder exited with code $LASTEXITCODE; restarting in 10 s (the downtime is logged as a gap)."
    Start-Sleep -Seconds 10
}

