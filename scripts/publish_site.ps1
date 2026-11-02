# Rebuilds the website from the databases and pushes it to GitHub Pages every 15 minutes. Stop with Ctrl+C.
# Run from the project folder in its own terminal, next to the recorder:
#   powershell -ExecutionPolicy Bypass -File scripts\publish_site.ps1
param([int]$Minutes = 15)

while ($true) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm 'UTC'")
    python -m dvd build-site
    if ($LASTEXITCODE -eq 0) {
        git add docs
        git diff --cached --quiet
        if ($LASTEXITCODE -ne 0) {
            git commit -q -m "Automatic site update $stamp"
            git push -q
            if ($LASTEXITCODE -eq 0) { Write-Host "$stamp published" } else { Write-Host "$stamp push failed; will retry next round" }
        } else { Write-Host "$stamp no changes" }
    } else { Write-Host "$stamp site build failed; will retry next round" }
    Start-Sleep -Seconds ($Minutes * 60)
}
