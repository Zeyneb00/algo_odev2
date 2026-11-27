"""
data_io.py  -  reading CSV / TXT files for the app

The assignment says files can be CSV or TXT, so this reader handles:
  * delimiters: comma, semicolon, tab, or plain spaces
  * files with a header row AND files without one (UEF .txt files)
  * decimal commas ("1,5") in semicolon files
  * a UTF-8 BOM, or Latin-1 encoded files
"""

import csv
import os


def _read_lines(path):
    for enc in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return [ln for ln in f.read().splitlines() if ln.strip()]
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode file")


def _detect_delimiter(line):
    if "\t" in line:
        return "\t"
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    return None   # whitespace


def _split(lines, delim):
    if delim is None:
        return [ln.split() for ln in lines]
    return [[c.strip() for c in row] for row in csv.reader(lines, delimiter=delim)]


def to_float(text, decimal_comma=False):
    """float(text) or None if blank / not a number."""
    t = text.strip()
    if not t:
        return None
    if decimal_comma:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def read_table(path):
    """
    Returns (header, rows, decimal_comma)
      header : list of column names
      rows   : list of lists of strings, every row as long as the header
      decimal_comma : True if numbers in this file use ',' as the decimal mark
    """
    lines = _read_lines(path)
    if not lines:
        raise ValueError("file is empty")

    delim = _detect_delimiter(lines[0])
    table = _split(lines, delim)
    decimal_comma = delim == ";"

    # header present if at least one cell of the first row is not a number
    first = table[0]
    has_header = any(to_float(c, decimal_comma) is None and c != "" for c in first)
    if has_header:
        header = [c if c else f"col_{i}" for i, c in enumerate(first)]
        body = table[1:]
    else:
        header = [f"col_{i}" for i in range(len(first))]
        body = table

    width = len(header)
    rows = []
    for r in body:
        r = r[:width] + [""] * (width - len(r))    # pad / trim ragged rows
        rows.append(r)
    return header, rows, decimal_comma


def numeric_columns(header, rows, decimal_comma=False):
    """Names of columns whose non-blank cells are all numbers (and not all blank)."""
    result = []
    for ci, name in enumerate(header):
        seen = 0
        ok = True
        for r in rows:
            cell = r[ci].strip()
            if not cell:
                continue
            if to_float(cell, decimal_comma) is None:
                ok = False
                break
            seen += 1
        if ok and seen > 0:
            result.append(name)
    return result


def extract_points(header, rows, col_x, col_y, decimal_comma=False):
    """
    Build the (x, y) points from two chosen columns.
    Rows where either value is blank / not numeric are skipped.
    Returns (points, skipped_count).
    """
    xi, yi = header.index(col_x), header.index(col_y)
    points, skipped = [], 0
    for r in rows:
        x = to_float(r[xi], decimal_comma)
        y = to_float(r[yi], decimal_comma)
        if x is None or y is None:
            skipped += 1
        else:
            points.append((x, y))
    return points, skipped


def manifest_columns(manifest_path, file_path):
    """Look up the prescribed (col_x, col_y) for a file name, or None."""
    if not os.path.exists(manifest_path):
        return None
    name = os.path.basename(file_path).lower()
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["file"].strip().lower() == name:
                return row["col_x"].strip(), row["col_y"].strip()
    return None
