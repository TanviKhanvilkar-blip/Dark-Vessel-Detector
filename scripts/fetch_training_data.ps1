# Fetches everything the CNN is trained on: for each Danish training strait, the coastline and structure masks
# (OpenStreetMap) and the Sentinel-1 scene crops (Planetary Computer) for days whose AIS is in train_ais.sqlite.
# The Oresund is deliberately absent: it is the held-out test area.
$pairs = @(
    @("greatbelt",  "S1D_IW_GRDH_1SDV_20261026T053204_20261026T053229_005182_009D5A"),
    @("littlebelt", "S1D_IW_GRDH_1SDV_20261026T053204_20261026T053229_005182_009D5A"),
    @("littlebelt", "S1D_IW_GRDH_1SDV_20261029T170847_20261029T170912_005233_009F0B"),
    @("fehmarn",    "S1D_IW_GRDH_1SDV_20261026T053229_20261026T053254_005182_009D5A"),
    @("fehmarn",    "S1C_IW_GRDH_1SDV_20261027T052403_20261027T052428_010066_01417A"),
    @("kattegat",   "S1D_IW_GRDH_1SDV_20261026T053139_20261026T053204_005182_009D5A"),
    @("skagen",     "S1D_IW_GRDH_1SDV_20261026T053139_20261026T053204_005182_009D5A")
)
foreach ($area in ($pairs | ForEach-Object { $_[0] } | Select-Object -Unique)) {
    foreach ($cmd in "fetch-coast", "fetch-infra") {
        for ($i = 1; $i -le 5; $i++) {
            python -m dvd $cmd --region $area
            if ($LASTEXITCODE -eq 0) { break }
            Write-Host "Overpass busy, retry $i"; Start-Sleep -Seconds 20
        }
    }
}
foreach ($p in $pairs) {
    $date = $p[1].Split("_")[4].Substring(0, 8)
    python -m dvd fetch-scene $p[1] --region $p[0] --out "scenes/train/$($p[0])_$date"
}
