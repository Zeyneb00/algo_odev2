from pathlib import Path
import pandas as pd

RAW_B = Path("datasets/raw_datasets_B")
CSV_B = Path("datasets/csv_datasets_B")
NOTES_PATH = CSV_B / "group_b_prescribed_pairs.txt"

DROP_COLUMNS = {
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
}

# Configure per-dataset entry when needed:
# rel_path: path relative to RAW_B (can be a file inside a subfolder, e.g. "Mall_Customers/Mall_Customers.csv")
# sep: ",", ";", "\t", r"\s+", or None (None used for .xlsx)
# has_header: True or False
# extra_drop: list of extra column names to drop
# prescribed_pair: tuple of (col_x, col_y) to record in the notes file
DATASETS_B: list[dict] = [
    # Example:
    # {
    #     "name": "mall_customers",
    #     "rel_path": "Mall_Customers/Mall_Customers.csv",
    #     "sep": ",",
    #     "has_header": True,
    #     "extra_drop": [],
    #     "prescribed_pair": ("Annual Income (k$)", "Spending Score (1-100)"),
    # },
]


def load_table(src: Path, sep: str | None, has_header: bool) -> pd.DataFrame:
    ext = src.suffix.lower()
    header_arg = 0 if has_header else None

    if ext == ".xlsx":
        df = pd.read_excel(src, engine="openpyxl", header=header_arg)
    else:
        effective_sep = sep if sep is not None else ","
        df = pd.read_csv(
            src,
            sep=effective_sep,
            engine="python" if effective_sep == r"\s+" or len(effective_sep) > 1 else "c",
            header=header_arg,
            skipinitialspace=True,
        )

    if not has_header:
        df.columns = [f"feature_{i}" for i in range(df.shape[1])]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    return df


def clean_group_b(df: pd.DataFrame, extra_drop: list[str] | None = None) -> pd.DataFrame:
    drop_set = {c.strip().lower() for c in DROP_COLUMNS}
    if extra_drop:
        drop_set |= {c.strip().lower() for c in extra_drop}

    kept_numeric: dict[str, pd.Series] = {}
    for col in df.columns:
        if col.strip().lower() in drop_set:
            continue
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        # Keep column if it has numeric values and is not predominantly non-numeric text
        non_null_raw = df[col].dropna()
        if len(non_null_raw) > 0 and numeric_series.notna().sum() == len(non_null_raw):
            kept_numeric[col] = numeric_series

    if not kept_numeric:
        raise ValueError("No valid numeric feature columns remained after filtering.")

    out_df = pd.DataFrame(kept_numeric).dropna().reset_index(drop=True)
    return out_df


def main() -> None:
    CSV_B.mkdir(parents=True, exist_ok=True)
    notes_lines: list[str] = ["Group B Prescribed Column Pairs", "=" * 32, ""]

    for item in DATASETS_B:
        src = RAW_B / item["rel_path"]
        raw_df = load_table(
            src=src,
            sep=item.get("sep"),
            has_header=item.get("has_header", True),
        )
        cleaned = clean_group_b(raw_df, extra_drop=item.get("extra_drop"))

        out_name = f"{item['name']}.csv"
        out_file = CSV_B / out_name
        cleaned.to_csv(out_file, sep=",", index=False)

        pair = item.get("prescribed_pair")
        pair_str = f"{pair[0]} , {pair[1]}" if pair else "TBD"
        notes_lines.append(f"{out_name}: {pair_str}  [columns: {list(cleaned.columns)}]")
        print(f"OK: {src} -> {out_file} ({cleaned.shape[0]} rows, {cleaned.shape[1]} numeric cols)")

    NOTES_PATH.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    print(f"Wrote notes to {NOTES_PATH}")


if __name__ == "__main__":
    main()