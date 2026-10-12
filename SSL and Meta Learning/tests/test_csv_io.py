import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

from datasets.io import load_image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _test_module():
    spec = importlib.util.spec_from_file_location("grader_test_py", os.path.join(REPO, "test.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_write_predictions_exact_bytes(tmp_path):
    t = _test_module()
    p = str(tmp_path / "predictions.csv")
    t.write_predictions(p, ["3", "1", "10"], ["/a/x.png", "/b/y, z.png", "/c/z.png"], np.array([0, 50, 7]))
    data = open(p, "rb").read()
    assert data == b'id,file_path,label\n3,/a/x.png,0\n1,"/b/y, z.png",50\n10,/c/z.png,7\n'


def test_read_csv_bom_order_extra_columns(tmp_path):
    t = _test_module()
    p = tmp_path / "t.csv"
    p.write_bytes("﻿id,file_path,extra\n5,/p/5.png,a\n2,/p/2.png,b\n\n9,/p/9.png,c\n".encode("utf-8"))
    ids, fps = t.read_test_csv(str(p))
    assert ids == ["5", "2", "9"] and fps == ["/p/5.png", "/p/2.png", "/p/9.png"]


def test_read_csv_one_row(tmp_path):
    t = _test_module()
    p = tmp_path / "t.csv"
    p.write_text("id,file_path\n0,/x.png\n")
    assert t.read_test_csv(str(p)) == (["0"], ["/x.png"])


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P", "LA", "I;16", "CMYK", "1"])
def test_load_image_modes(tmp_path, mode):
    rng = np.random.RandomState(0)
    rgb = Image.fromarray(rng.randint(0, 255, (37, 53, 3), dtype=np.uint8))
    if mode == "I;16":
        im = Image.fromarray((np.asarray(rgb.convert("L"), np.uint16) * 257))
    elif mode == "P":
        im = rgb.convert("P", palette=Image.ADAPTIVE)
    else:
        im = rgb.convert(mode)
    path = str(tmp_path / f"img.{'jpg' if mode == 'CMYK' else 'png'}")
    im.save(path)
    a = load_image(path, 24, 3)
    assert a.shape == (3, 24, 24) and a.dtype == np.uint8


def test_16bit_is_rescaled_not_clipped(tmp_path):
    arr = np.full((20, 20), 30000, np.uint16)      # ~117/255 after correct rescaling
    path = str(tmp_path / "x.png")
    Image.fromarray(arr).save(path)
    a = load_image(path, 16, 3)
    assert 100 < a.mean() < 130


def test_corrupt_raises(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n garbage")
    with pytest.raises(Exception):
        load_image(str(p), 16, 3)
