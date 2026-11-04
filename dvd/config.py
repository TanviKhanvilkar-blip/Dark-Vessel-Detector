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

# Training areas for the ship/not-ship CNN: Danish straits fully covered by Sentinel-1 on days we hold AIS for.
# The Oresund ("denmark") is deliberately NOT here: it is the held-out test area.
TRAIN_AREAS = {
    "greatbelt": (10.85, 55.20, 11.25, 55.50),
    "littlebelt": (9.65, 55.45, 10.05, 55.75),
    "fehmarn": (11.05, 54.45, 11.45, 54.75),
    "kattegat": (11.30, 56.55, 11.70, 56.85),
    "skagen": (10.40, 57.55, 10.80, 57.85),
    # offshore wind farms: the first model mistook turbines for ships (87% on the test set), so it needs more examples
    "rodsand": (11.40, 54.45, 11.80, 54.65),     # Nysted + Rodsand II wind farms
    "anholt": (11.05, 56.50, 11.40, 56.75),      # Anholt wind farm
}
ALL_AREAS = {**REGIONS, **TRAIN_AREAS}

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
CNN_THRESHOLD = 0.5        # ShipCNN P(ship) below this filters an unmatched target as "not a ship"
# An untracked target in the same place on another pass is probably fixed (unmapped structure, mooring, permanently
# anchored hulk), not a ship going dark. Radius allows for geolocation error between passes (~150 m) plus swinging.
REPEAT_RADIUS_M = 600.0

# Matching
MATCH_WINDOW_S = 1800      # use AIS within +/- 30 min of the image time
MATCH_BASE_M = 100.0
MATCH_SPEED_FACTOR = 0.15  # extra radius per metre of distance the ship could have travelled in dt
# SAR azimuth shift: a ship moving toward/away from the satellite is imaged displaced along the satellite track by
# (slant range / satellite speed) x (its velocity along the look direction). For Sentinel-1 IW that ratio is ~110 s,
# so 10 kn can mean ~500 m. Without orbit geometry we allow the worst case (all of the speed along the look direction).
SAR_RANGE_OVER_VELOCITY_S = 110.0

# Size sanity for matches. Checking matched pairs against ships' real AIS lengths showed small moored boats claiming the
# radar return of the quay beside them (median 16 m boats on 91 m targets). So a pair is refused when the target is far
# too long for the ship, and small craft at a quay cannot match at all: they can't be told apart from the quay.
MATCH_MAX_LENGTH_RATIO = 2.5
MATCH_LENGTH_SLACK_M = 40.0        # radar smears small targets, so a real tug of 24 m can measure 44 m
QUAY_SHORE_KM = 0.3
QUAY_MIN_SHIP_M = 40.0             # same limit as MIN_LENGTH_M: what the satellite can reliably resolve
SMALL_CRAFT_TYPES = ("Pleasure", "Sailing")
RADAR_LENGTH_BIAS = 1.7            # radar length / AIS length: median over the 20 matched Oresund ships (moving ones ~2, moored ~1.2)

# AIS coverage honesty: free ground receivers only reach so far offshore (approximate).
RX_RANGE_KM = 18.0
MAX_AIS_GAP_S = 300        # a feed gap longer than this inside the window makes all flags low confidence

# AIS recorder
AIS_PAD_DEG = 0.15         # record ~16 km beyond the AOI so ships entering during the +/-30 min window are covered
AIS_GAP_THRESHOLD_S = 60   # silence longer than this is logged as a gap (busy port: normal spacing is seconds)
AIS_STALL_S = 180          # connected but silent this long -> reconnect

KNOT_MS = 0.514444
