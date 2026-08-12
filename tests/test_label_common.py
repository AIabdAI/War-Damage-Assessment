# tests/test_label_common.py — locks the canonical parse contract shared by
# prepare_split and crop_classification (object alignment depends on it).
from pathlib import Path

from scripts.label_common import index_images, parse_label_file


def write(tmp_path: Path, text: str, name: str = "sample.txt") -> Path:
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


def test_valid_line(tmp_path):
    objs, bad, clamped = parse_label_file(
        write(tmp_path, "3 0.5 0.5 0.2 0.3 1\n"), num_classes=11)
    assert len(objs) == 1 and not bad and not clamped
    o = objs[0]
    assert (o.line, o.cls, o.flag) == (1, 3, 1)
    assert (o.cx, o.cy, o.w, o.h) == (0.5, 0.5, 0.2, 0.3)


def test_skip_reasons(tmp_path):
    text = ("3 0.5 0.5 0.2 0.3\n"          # 5 tokens -> missing_damage_flag
            "3 0.5 0.5 0.2 0.3 1 9\n"      # 7 tokens -> bad_columns
            "x 0.5 0.5 0.2 0.3 1\n"        # non_numeric
            "11 0.5 0.5 0.2 0.3 1\n"       # cls >= num_classes
            "-1 0.5 0.5 0.2 0.3 1\n"       # negative cls -> old_scheme_remnant
            "3 0.5 0.5 0.2 0.3 2\n")       # invalid_damage_flag
    objs, bad, _ = parse_label_file(write(tmp_path, text), num_classes=11)
    assert not objs
    assert [b["reason"] for b in bad] == [
        "missing_damage_flag", "bad_columns", "non_numeric",
        "old_scheme_remnant", "old_scheme_remnant", "invalid_damage_flag"]
    assert [b["line"] for b in bad] == [1, 2, 3, 4, 5, 6]


def test_clamp_only_when_coord_out_of_range(tmp_path):
    # w > 1: coordinate itself out of range -> clamped + recorded
    objs, _, clamped = parse_label_file(
        write(tmp_path, "0 0.5 0.5 1.2 0.4 0\n"), num_classes=11)
    assert clamped == [{"file": "sample.txt", "line": 1}]
    assert objs[0].w <= 1.0
    # edge overflow with in-range coords (cx=0.05, w=0.2 -> left < 0):
    # NOT clamped — detection labels and crops must both keep the original
    objs, _, clamped = parse_label_file(
        write(tmp_path, "0 0.05 0.5 0.2 0.4 0\n"), num_classes=11)
    assert not clamped
    assert (objs[0].cx, objs[0].w) == (0.05, 0.2)


def test_crlf_line_numbering_stable(tmp_path):
    # read_text's universal newlines turn CRLF into LF before parsing, so a
    # CRLF file numbers 1,2 — the point is that BOTH consumers share this
    # exact numbering (it names the crop files)
    objs, _, _ = parse_label_file(
        write(tmp_path, "0 0.5 0.5 0.2 0.2 0\r\n1 0.5 0.5 0.2 0.2 1\r\n"),
        num_classes=11)
    assert [o.line for o in objs] == [1, 2]


def test_form_feed_is_one_line_not_two(tmp_path):
    # \x0c is NOT a line separator here (unlike str.splitlines): two records
    # joined by a form feed parse as ONE bad_columns line, keeping numbering
    # identical in both consumers
    objs, bad, _ = parse_label_file(
        write(tmp_path, "0 0.5 0.5 0.2 0.2 0\x0c1 0.5 0.5 0.2 0.2 1\n"),
        num_classes=11)
    assert not objs
    assert [(b["line"], b["reason"]) for b in bad] == [(1, "bad_columns")]


def test_empty_file_is_valid_negative(tmp_path):
    objs, bad, clamped = parse_label_file(write(tmp_path, ""), num_classes=11)
    assert objs == [] and bad == [] and clamped == []


def test_index_images_prefers_ext_priority(tmp_path):
    (tmp_path / "x.jpeg").write_bytes(b"a")   # sorts before x.jpg by name
    (tmp_path / "x.jpg").write_bytes(b"b")
    (tmp_path / "y.PNG").write_bytes(b"c")    # case-insensitive match
    (tmp_path / ".hidden.jpg").write_bytes(b"d")
    idx = index_images(tmp_path)
    assert idx["x"].name == "x.jpg"           # .jpg beats .jpeg despite sort order
    assert idx["y"].name == "y.PNG"
    assert len(idx) == 2
