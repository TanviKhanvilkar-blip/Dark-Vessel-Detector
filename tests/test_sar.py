import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin

from dvd import config as C
from dvd.detect import detect_scene
from dvd.sar import load_scene, time_from_name

N = 3300  # a bit larger than the 3000 px AOI so there is margin


def _dn_image(rng):
    img = np.sqrt(rng.gamma(4.0, 0.25, size=(N, N))) * 100          # amplitude DN, calm sea
    img[1500:1503, 1000:1006] *= 8                                   # one ship-like bright blob
    return np.clip(img, 1, 65535).astype(np.uint16)


def _write(path, gcp_mode):
    rng = np.random.default_rng(0)
    res = C.RES_DEG
    west, north = C.AOI[0] - 0.01, C.AOI[3] + 0.01
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=1, dtype="uint16",
                       **({} if gcp_mode else dict(crs="EPSG:4326", transform=from_origin(west, north, res, res)))) as ds:
        ds.write(_dn_image(rng), 1)
        if gcp_mode:
            pts = [(r, c) for r in (0, N) for c in (0, N)]
            ds.gcps = ([GroundControlPoint(r, c, west + c * res, north - r * res) for r, c in pts], "EPSG:4326")


@pytest.mark.parametrize("gcp_mode,method", [(False, "tps"), (True, "tps"), (True, "polynomial")])
def test_load_scene_places_and_detects_ship(tmp_path, gcp_mode, method):
    p = str(tmp_path / "s1a-iw-grd-vv-20260104t013200-x.tiff")
    _write(p, gcp_mode)
    sc = load_scene(p, gcp_method=method)
    assert sc.image.shape == (sc.grid.height, sc.grid.width)
    assert 0.8 < np.median(sc.image) < 1.2                           # normalised to ~1
    det = detect_scene(sc.image, sc.land, sc.grid)
    assert len(det) == 1
    # blob centre in the file is row 1501, col 1003.5; file origin is 0.01 deg west/north of the AOI
    lon = C.AOI[0] - 0.01 + 1003.5 * C.RES_DEG
    lat = C.AOI[3] + 0.01 - 1501 * C.RES_DEG
    assert abs(det.lon[0] - lon) < 3 * C.RES_DEG and abs(det.lat[0] - lat) < 3 * C.RES_DEG


def test_calibration_lut_interpolates(tmp_path):
    from rasterio.windows import Window
    from dvd.sar import sigma0_lut
    xml = ("<calibration><calibrationVectorList>"
           "<calibrationVector><line>0</line><pixel count='2'>0 100</pixel><sigmaNought count='2'>100 200</sigmaNought></calibrationVector>"
           "<calibrationVector><line>100</line><pixel count='2'>0 100</pixel><sigmaNought count='2'>300 400</sigmaNought></calibrationVector>"
           "</calibrationVectorList></calibration>")
    p = tmp_path / "cal.xml"
    p.write_text(xml)
    lut = sigma0_lut(str(p), Window(0, 0, 101, 101))
    assert lut[0, 0] == 100 and lut[0, 100] == 200 and lut[100, 0] == 300 and abs(lut[50, 50] - 250) < 1e-3


def test_time_from_name():
    assert time_from_name("S1A_IW_GRDH_1SDV_20260104T013200_x.tiff") == 1767490320.0


def test_aoi_outside_scene_raises(tmp_path):
    p = str(tmp_path / "s1a-20260104t013200.tiff")
    _write(p, False)
    with pytest.raises(ValueError):
        load_scene(p, aoi=(10.0, 10.0, 10.3, 10.3))
