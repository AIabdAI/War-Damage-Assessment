# tests/test_labels.py
from pathlib import Path

from verify_labels import Stats, validate_file


def write_label(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_line(tmp_path):
    issues = validate_file(write_label(tmp_path, "3 0.5 0.5 0.2 0.3 1\n"),
                           Stats(), num_classes=12)
    assert not issues.has_errors
    assert not issues.is_empty


def test_class_out_of_range(tmp_path):
    issues = validate_file(write_label(tmp_path, "13 0.5 0.5 0.2 0.3 0\n"),
                           Stats(), num_classes=12)
    assert issues.range_errors
    assert "class_id 13" in issues.range_errors[0][1]


def test_wh_exceeds_one(tmp_path):
    issues = validate_file(write_label(tmp_path, "2 0.5 0.5 1.4 0.3 0\n"),
                           Stats(), num_classes=12)
    assert any("w=1.4" in msg for _, msg in issues.range_errors)


def test_wrong_column_count(tmp_path):
    issues = validate_file(write_label(tmp_path, "2 0.5 0.5 0.2\n"),
                           Stats(), num_classes=12)
    assert issues.bad_col_lines == [(1, 4)]


def test_bad_damage_flag(tmp_path):
    issues = validate_file(write_label(tmp_path, "2 0.5 0.5 0.2 0.3 7\n"),
                           Stats(), num_classes=12)
    assert any("damage_flag 7" in msg for _, msg in issues.range_errors)


def test_empty_file_is_valid_negative(tmp_path):
    stats = Stats()
    issues = validate_file(write_label(tmp_path, ""), stats, num_classes=12)
    assert issues.is_empty
    assert not issues.has_errors
    assert stats.empty_files == 1
