from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class IndicatorFrame:
    """Container for a single indicator time series."""

    workbook: str
    sheet: str
    indicator_id: str
    indicator_name: str
    unit: str
    frame: pd.DataFrame


def slugify(label: str) -> str:
    """Create a filesystem-friendly identifier."""

    slug = re.sub(r"[^0-9A-Za-z]+", "-", label)
    slug = re.sub(r"-{2,}", "-", slug).strip("-").lower()
    return slug or "indicator"


def split_indicator_label(label: Optional[str]) -> Tuple[str, str]:
    """Split the indicator label into name/unit parts."""

    if not isinstance(label, str):
        return "Unknown indicator", ""
    parts = [part.strip() for part in label.split(",") if part and part.strip()]
    if not parts:
        return "Unknown indicator", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], ", ".join(parts[1:])


def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    """Return the index of the row that stores the 'State' header."""

    first_col = df.iloc[:, 0].astype(str).str.strip().str.lower()
    matches = df.index[first_col == "state"]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _normalize_header_cells(raw_cells: Iterable) -> List[Optional[str]]:
    normalized: List[Optional[str]] = []
    for idx, cell in enumerate(raw_cells):
        if isinstance(cell, str):
            cell = cell.strip()
        if pd.isna(cell):
            normalized.append(None)
            continue
        if isinstance(cell, (int, float)) and not isinstance(cell, bool):
            if math.isnan(cell):
                normalized.append(None)
            else:
                normalized.append(str(int(cell)))
            continue
        normalized.append(str(cell))
    return normalized


def _tidy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide sheet (states x years) to a tidy frame."""

    header_idx = _find_header_row(df)
    if header_idx is None:
        raise ValueError("Header row with 'State' not found")

    header_cells = _normalize_header_cells(df.iloc[header_idx].tolist())
    valid_positions = [i for i, val in enumerate(header_cells) if val]
    df_data = df.iloc[header_idx + 1 :, valid_positions].copy()
    df_data.columns = [header_cells[i] for i in valid_positions]
    if "State" not in df_data.columns:
        raise ValueError("Normalized header does not include 'State'")

    df_data = df_data[df_data["State"].notna()].copy()
    df_data["State"] = (
        df_data["State"].astype(str).str.strip().replace({"": pd.NA})
    )
    df_data = df_data[df_data["State"].notna()]

    value_columns = [col for col in df_data.columns if col != "State"]
    tidy = df_data.melt(
        id_vars="State", value_vars=value_columns, var_name="Year", value_name="value"
    )
    tidy["Year"] = pd.to_numeric(tidy["Year"], errors="coerce").astype("Int64")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["State", "Year", "value"])
    tidy["Year"] = tidy["Year"].astype(int)
    tidy["State"] = tidy["State"].str.upper()
    return tidy


def _read_indicator(file_path: Path, sheet_name: str) -> IndicatorFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    indicator_label = str(df.iloc[0, 0]) if not df.empty else file_path.stem
    indicator_name, unit = split_indicator_label(indicator_label)
    tidy = _tidy_sheet(df)
    indicator_id = slugify(f"{file_path.stem}-{sheet_name}")
    tidy = tidy.rename(columns={"State": "region", "Year": "year"})
    tidy.insert(0, "indicator_id", indicator_id)
    return IndicatorFrame(
        workbook=file_path.name,
        sheet=sheet_name,
        indicator_id=indicator_id,
        indicator_name=indicator_name,
        unit=unit,
        frame=tidy,
    )


def load_all_indicators(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all Excel files in data_dir and return (long data, metadata)."""

    records: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, object]] = []

    for workbook in sorted(data_dir.glob("*.xlsx")):
        xl = pd.ExcelFile(workbook)
        for sheet_name in xl.sheet_names:
            if sheet_name.lower() == "contents":
                continue
            try:
                indicator = _read_indicator(workbook, sheet_name)
            except Exception as exc:  # pragma: no cover - diagnostic path
                LOGGER.warning("Skip %s - %s (%s)", workbook.name, sheet_name, exc)
                continue
            records.append(indicator.frame)
            metadata_rows.append(
                {
                    "indicator_id": indicator.indicator_id,
                    "indicator_name": indicator.indicator_name,
                    "unit": indicator.unit,
                    "workbook": indicator.workbook,
                    "sheet": indicator.sheet,
                    "min_year": indicator.frame["year"].min(),
                    "max_year": indicator.frame["year"].max(),
                    "observations": len(indicator.frame),
                }
            )
    combined = pd.concat(records, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows).sort_values("indicator_id").reset_index(
        drop=True
    )
    return combined, metadata


def export_data(
    long_frame: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "indicator_timeseries.csv"
    meta_path = output_dir / "indicator_metadata.csv"
    long_frame.to_csv(long_path, index=False)
    metadata.to_csv(meta_path, index=False)
    try:
        import pyarrow  # type: ignore

        long_frame.to_parquet(output_dir / "indicator_timeseries.parquet", index=False)
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.info("pyarrow not available, parquet export skipped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw SEDS workbooks into a tidy indicator dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the raw Excel files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/intermediate"),
        help="Destination for cleaned CSV/metadata files.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    long_frame, metadata = load_all_indicators(args.data_dir)
    export_data(long_frame, metadata, args.output_dir)
    LOGGER.info(
        "Cleaned %d indicators with %d records",
        metadata.shape[0],
        long_frame.shape[0],
    )


if __name__ == "__main__":
    main()
