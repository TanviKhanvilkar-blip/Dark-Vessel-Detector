"""Central settings. Every threshold the pipeline uses lives here so it can be tuned in one place."""

# Area of interest: (west, south, east, north) in degrees. Roughly the sea west of Mumbai.
AOI = (72.45, 18.80, 72.75, 19.10)
RES_DEG = 0.0001  # ~11 m per pixel, close to Sentinel-1 GRD at 10 m
# Areas used for real data (scenes, AIS recording, coastline, structures). AOI above stays the synthetic demo area.
# Mumbai: the project's target, ~1 Sentinel-1 pass every 12 days, sparse free AIS.
# Denmark (Oresund, Copenhagen-Malmo): ~daily Sentinel-1 passes, fully inside most swaths, and dense coastal AIS
# receivers, so matching can be shown end to end. (The Great Belt was tried first but sits on swath edges.)
REGIONS = {
    "mumbai": (72.55, 18.80, 72.95, 19.10),
    "denmark": (12.55, 55.55, 12.95, 55.85),
}
LIVE_AOI = REGIONS["mumbai"]

# Fixed structures to ignore (lon, lat). PLACEHOLDERS: replace with real platform/buoy positions.
FIXED_STRUCTURES = [(72.52, 18.93), (72.58, 19.04)]
STRUCTURE_RADIUS_M = 150.0       # around point structures: platforms, buoys, beacons
STRUCTURE_LINE_BUFFER_M = 120.0  # around line structures: bridges (pylons under the centreline), piers, breakwaters

# Detection
CFAR_THRESHOLD = 5.5       # pixel must exceed this multiple of local sea background
CFAR_GUARD_PX = 41         # guard window: wider than the longest ship so it does not pollute the background
CFAR_BG_PX = 101           # background window
MIN_AREA_PX = 4
TILE_CORE_PX = 512
TILE_HALO_PX = 64          # context around each tile so the CFAR background window never sees a tile edge
LAND_BUFFER_PX = 3

# Alert filters
MIN_SCORE = 0.60
MIN_LENGTH_M = 40.0        # smaller ships are not reliably visible; stated limit of the system
MIN_SHORE_KM = 0.5

# Matching
MATCH_WINDOW_S = 1800      # use AIS within +/- 30 min of the image time
MATCH_BASE_M = 100.0
MATCH_SPEED_FACTOR = 0.15  # extra radius per metre of distance the ship could have travelled in dt
# SAR azimuth shift: a ship moving toward/away from the satellite is imaged displaced along the satellite track by
# (slant range / satellite speed) x (its velocity along the look direction). For Sentinel-1 IW that ratio is ~110 s,
# so 10 kn can mean ~500 m. Without orbit geometry we allow the worst case (all of the speed along the look direction).
SAR_RANGE_OVER_VELOCITY_S = 110.0

# AIS coverage honesty: free ground receivers only reach so far offshore (approximate).
RX_RANGE_KM = 18.0
MAX_AIS_GAP_S = 300        # a feed gap longer than this inside the window makes all flags low confidence

# AIS recorder
AIS_PAD_DEG = 0.15         # record ~16 km beyond the AOI so ships entering during the +/-30 min window are covered
AIS_GAP_THRESHOLD_S = 60   # silence longer than this is logged as a gap (busy port: normal spacing is seconds)
AIS_STALL_S = 180          # connected but silent this long -> reconnect

KNOT_MS = 0.514444
