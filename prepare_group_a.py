from pathlib import Path
import pandas as pd

RAW_B = Path("datasets/raw_datasets_B")
CSV_B = Path("datasets/csv_datasets_B")
NOTES_PATH = CSV_B / "group_b_prescribed_pairs.txt"

DROP_EXACT = {
    "id",
    "customerid",
    "species",
    "class",
    "label",
    "target",
    "channel",
    "region",
    "genre",
    "gender",
    "type",  # e.g. Facebook post Type (Photo, Status, Link, Video)
}

# Optional per-stem overrides (keyed by filename without extension, lowercased):
# - sep: ",", ";", "\t", r"\s+", or None (for .xlsx)
# - has_header: True or False
# - drop_indices: list of 0-based column positions to drop (handy for headerless files)
# - extra_drop: list of column names to drop (case-insensitive)
# - prescribed_pair: (col_x, col_y) to write into group_b_prescribed_pairs.txt
FILE_CONFIGS: dict[str, dict] = {
    "dataset_facebook": {
        "sep": ";",
        "has_header": True,
        "prescribed_pair": None,
    },
    "iris": {
        "sep": ",",
        "has_header": False,
        "drop_indices": [4],  # 5th column is species
        "prescribed_pair": None,
    },
    "raisin_dataset": {
        "sep": None,
        "has_header": True,
        "prescribed_pair": None,
    },
    "seeds_dataset": {
        "sep": r"\s+",
        "has_header": False,
        "drop_indices": [7],  # 8th column is seed variety class
        "prescribed_pair": None,
    },
    "wholesale customers data": {
        "sep": ",",
        "has_header": True,
        "extra_drop": ["Channel", "Region"],
        "prescribed_pair": None,
    },
}


def detect_sep_and_header(path: Path) -> tuple[str, bool]:
    lines = [
        line.strip("\r\n")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ][:5]
    sample = "\n".join(lines)

    if ";" in sample:
        sep = ";"
    elif "\t" in sample:
        sep = "\t"
    elif "," in sample:
        sep = ","
    else:
        sep = r"\s+"

    first_tokens = lines[0].split(sep if len(sep) == 1 else None)
    numeric_count = 0
    for tok in first_tokens:
        try:
            float(tok.strip())
            numeric_count += 1
        except ValueError:
            pass

    # Treat as headerless if most/all first-row tokens are numeric
    has_header = numeric_count < max(1, len(first_tokens) // 2 + 1)
    return sep, has_header


def load_raw(path: Path, cfg: dict) -> pd.DataFrame:
    ext = path.suffix.lower()

    if ext == ".xlsx":
        has_header = cfg.get("has_header", True)
        df = pd.read_excel(path, engine="openpyxl", header=0 if has_header else None)
    else:
        auto_sep, auto_header = detect_sep_and_header(path)
        sep = cfg.get("sep", auto_sep)
        has_header = cfg.get("has_header", auto_header)
        df = pd.read_csv(
            path,
            sep=sep,
            engine="python" if sep == r"\s+" or len(sep) > 1 else "c",
            header=0 if has_header else None,
            skipinitialspace=True,
        )

    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    return df


def clean_numeric_keep_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    work = df.copy()

    drop_indices = cfg.get("drop_indices") or []
    valid_indices = [work.columns[i] for i in drop_indices if 0 <= i < work.shape[1]]
    if valid_indices:
        work = work.drop(columns=valid_indices)

    drop_names = {c.strip().lower() for c in DROP_EXACT}
    for extra in cfg.get("extra_drop") or []:
        drop_names.add(str(extra).strip().lower())

    kept: dict[str, pd.Series] = {}
    for col in work.columns:
        if str(col).strip().lower() in drop_names:
            continue
        # Skip obvious ID columns like "id" or ending with "_id" / "id"
        col_lower = str(col).strip().lower()
        if col_lower.endswith("_id") or col_lower.endswith("id") and col_lower != "paid":
            continue

        s_num = pd.to_numeric(work[col], errors="coerce")
        non_null = work[col].dropna()
        # Keep column if all non-null entries parse as numbers
        if len(non_null) > 0 and s_num.notna().sum() == len(non_null):
            kept[str(col)] = s_num

    if not kept:
        raise ValueError("No numeric columns remained after filtering.")

    return pd.DataFrame(kept).dropna().reset_index(drop=True)


def main() -> None:
    if not RAW_B.exists():
        raise FileNotFoundError(f"Input folder not found: {RAW_B.resolve()}")

    supported_exts = {".csv", ".txt", ".data", ".xlsx"}
    files = sorted(
        p for p in RAW_B.iterdir() if p.is_file() and p.suffix.lower() in supported_exts
    )
    if not files:
        print(f"No supported data files found in {RAW_B}")
        return

    CSV_B.mkdir(parents=True, exist_ok=True)
    notes_lines = ["Group B Prescribed Column Pairs", "=" * 32, ""]

    for src in files:
        key = src.stem.lower()
        cfg = FILE_CONFIGS.get(key, {})
        raw_df = load_raw(src, cfg)
        cleaned = clean_numeric_keep_all(raw_df, cfg)

        safe_stem = "_".join(src.stem.strip().split())
        out_csv = CSV_B / f"{safe_stem}.csv"
        cleaned.to_csv(out_csv, sep=",", index=False)

        pair = cfg.get("prescribed_pair")
        pair_text = f"{pair[0]}, {pair[1]}" if pair and len(pair) == 2 else "TBD"
        notes_lines.append(
            f"{out_csv.name}: {pair_text} | available_numeric_cols={list(cleaned.columns)}"
        )
        print(
            f"OK: {src.name} -> {out_csv} "
            f"({cleaned.shape[0]} rows, {cleaned.shape[1]} numeric cols)"
        )

    NOTES_PATH.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    print(f"Wrote notes to {NOTES_PATH}")


if __name__ == "__main__":
    main()