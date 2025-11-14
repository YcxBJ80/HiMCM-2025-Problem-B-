from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
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


ENVIRONMENT_WORKBOOK_NAME = "副本环境数据汇总.xlsx"
# 这些环境外部数据只提供最近年度的截面值，统一映射到2023年
ENVIRONMENT_DEFAULT_YEAR = 2023
STATE_WASTE_WORKBOOK_NAME = "State Waste Data.xlsx"
VALID_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}
CITY_WATER_COLUMN_METADATA = {
    "PS-Wtotl": {
        "name": "City water withdrawals – public supply",
        "unit": "million gallons per day (MGD)",
    },
    "IR-IrTot": {
        "name": "City water withdrawals – irrigation",
        "unit": "million gallons per day (MGD)",
    },
    "TO-Wtotl": {
        "name": "City water withdrawals – total withdrawals",
        "unit": "million gallons per day (MGD)",
    },
    "PT-Wtotl": {
        "name": "City water withdrawals – thermoelectric power",
        "unit": "million gallons per day (MGD)",
    },
    "IN-Wtotl": {
        "name": "City water withdrawals – industrial",
        "unit": "million gallons per day (MGD)",
    },
}
AIR_TRAVEL_COLUMN_METADATA = {
    "cy23_enplanements": {
        "name": "Commercial enplanements (CY 2023)",
        "unit": "passengers",
    },
    "cy22_enplanements": {
        "name": "Commercial enplanements (CY 2022 baseline)",
        "unit": "passengers",
    },
    "enplanement_growth": {
        "name": "Commercial enplanement growth rate (2023 vs 2022)",
        "unit": "share",
    },
    "airport_count": {
        "name": "Number of commercial airports with scheduled service",
        "unit": "count",
    },
}
STATE_NAME_TO_CODE = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "washington dc": "DC",
    "washington d.c": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "tenessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}
STATE_WASTE_COLUMN_METADATA = {
    "Recycling Rate if reported/calculatable": {
        "name": "Municipal solid waste recycling rate",
        "unit": "share of total MSW",
    },
    "Total MSW Generated (tons)": {
        "name": "Municipal solid waste generated",
        "unit": "short tons",
    },
    "Landfilled/Disposed MSW (tons)": {
        "name": "Municipal solid waste landfilled or disposed",
        "unit": "short tons",
    },
    "Recycled MSW (tons)": {
        "name": "Municipal solid waste recycled",
        "unit": "short tons",
    },
}
SPORTS_FACTORS_METADATA = {
    "nfl_stadium_capacity": {
        "name": "NFL stadium average capacity",
        "unit": "seats",
    },
    "stadium_hotel_distance_avg": {
        "name": "Average distance from NFL stadium to nearest hotels",
        "unit": "meters",
    },
    "peak_public_transport_capacity": {
        "name": "Peak public transportation capacity",
        "unit": "passengers per hour",
    },
    "avg_driving_time_to_nfl_states": {
        "name": "Average driving time to other NFL states",
        "unit": "seconds",
    },
    "avg_flight_time_to_nfl_states": {
        "name": "Average flight time to other NFL states",
        "unit": "seconds",
    },
}
SUPPLEMENTAL_INDICATOR_PREFIXES = (
    "city-water-use-cbsa",
    "airtravel-cy2023",
    "state-waste-data",
    "sports-factors",
)


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


def _extract_state_codes(title: str) -> List[str]:
    if not isinstance(title, str):
        return []
    if "," in title:
        suffix = title.split(",")[-1]
    else:
        suffix = title
    suffix = (
        suffix.replace(".", "")
        .replace("–", "-")
        .replace("—", "-")
        .replace("/", "-")
        .strip()
    )
    suffix = re.sub(r"\s+", "", suffix)
    if not suffix:
        return []
    codes = []
    for token in suffix.split("-"):
        token = token.strip().upper()
        if token in VALID_STATE_CODES:
            codes.append(token)
    return codes


def _state_name_to_code(name: object) -> Optional[str]:
    """Map a state name or code-like value to a two-letter state code."""

    if not isinstance(name, str):
        return None
    normalized = (
        name.strip()
        .replace(".", "")
        .replace("–", " ")
        .replace("—", " ")
        .replace("-", " ")
    )
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return None
    upper = normalized.upper()
    if len(upper) == 2 and upper in VALID_STATE_CODES:
        return upper
    lowered = normalized.lower()
    code = STATE_NAME_TO_CODE.get(lowered)
    if code:
        return code
    lowered_alpha = re.sub(r"[^a-z ]", " ", lowered)
    lowered_alpha = re.sub(r"\s+", " ", lowered_alpha).strip()
    return STATE_NAME_TO_CODE.get(lowered_alpha)


def _load_city_water_indicators(workbook: Path) -> List[IndicatorFrame]:
    sheet_name = "City_Water_Use_CBSA"
    try:
        df = pd.read_excel(workbook, sheet_name=sheet_name)
    except ValueError:
        LOGGER.warning("Worksheet %s missing in %s", sheet_name, workbook.name)
        return []
    required_cols = ["cbsatitle", *CITY_WATER_COLUMN_METADATA.keys()]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        LOGGER.warning(
            "Skip %s - %s missing columns %s", workbook.name, sheet_name, ", ".join(missing)
        )
        return []
    df = df.dropna(subset=["cbsatitle"]).copy()
    df["state_codes"] = df["cbsatitle"].apply(_extract_state_codes)
    df["state_count"] = df["state_codes"].apply(len)
    df = df[df["state_count"] > 0].copy()
    if df.empty:
        return []
    df["weight"] = 1.0 / df["state_count"]
    expanded = df.loc[:, ["state_codes", "weight"] + list(CITY_WATER_COLUMN_METADATA.keys())].copy()
    expanded = expanded.explode("state_codes")
    expanded = expanded.rename(columns={"state_codes": "region"})
    for col in CITY_WATER_COLUMN_METADATA.keys():
        expanded[col] = pd.to_numeric(expanded[col], errors="coerce") * expanded["weight"]
    grouped = (
        expanded.groupby("region")[list(CITY_WATER_COLUMN_METADATA.keys())]
        .sum()
        .reset_index()
    )
    frames: List[IndicatorFrame] = []
    for column, meta in CITY_WATER_COLUMN_METADATA.items():
        series = grouped[["region", column]].dropna(subset=[column])
        if series.empty:
            continue
        indicator_id = slugify(f"{workbook.stem}-{sheet_name}-{column}")
        tidy = pd.DataFrame(
            {
                "indicator_id": indicator_id,
                "region": series["region"],
                "year": ENVIRONMENT_DEFAULT_YEAR,
                "value": series[column],
            }
        )
        frames.append(
            IndicatorFrame(
                workbook=workbook.name,
                sheet=sheet_name,
                indicator_id=indicator_id,
                indicator_name=meta["name"],
                unit=meta["unit"],
                frame=tidy,
            )
        )
    return frames


def _load_air_travel_indicators(workbook: Path) -> List[IndicatorFrame]:
    sheet_name = "AirTravel_CY2023"
    try:
        df = pd.read_excel(workbook, sheet_name=sheet_name)
    except ValueError:
        LOGGER.warning("Worksheet %s missing in %s", sheet_name, workbook.name)
        return []
    required_cols = ["ST", "CY 23 Enplanements", "CY 22 Enplanements"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        LOGGER.warning(
            "Skip %s - %s missing columns %s", workbook.name, sheet_name, ", ".join(missing)
        )
        return []
    df = df.dropna(subset=["ST"]).copy()
    df["state"] = df["ST"].astype(str).str.strip().str.upper()
    df = df[df["state"].isin(VALID_STATE_CODES)].copy()
    if df.empty:
        return []
    df["cy23"] = pd.to_numeric(df["CY 23 Enplanements"], errors="coerce").fillna(0)
    df["cy22"] = pd.to_numeric(df["CY 22 Enplanements"], errors="coerce").fillna(0)
    state_group = df.groupby("state").agg(
        cy23_enplanements=("cy23", "sum"),
        cy22_enplanements=("cy22", "sum"),
        airport_count=("state", "size"),
    )
    state_group["enplanement_growth"] = np.where(
        state_group["cy22_enplanements"] > 0,
        (state_group["cy23_enplanements"] - state_group["cy22_enplanements"])
        / state_group["cy22_enplanements"],
        np.nan,
    )
    state_group = state_group.reset_index().rename(columns={"state": "region"})
    frames: List[IndicatorFrame] = []
    for column, meta in AIR_TRAVEL_COLUMN_METADATA.items():
        series = state_group[["region", column]].dropna(subset=[column])
        if series.empty:
            continue
        indicator_id = slugify(f"{workbook.stem}-{sheet_name}-{column}")
        tidy = pd.DataFrame(
            {
                "indicator_id": indicator_id,
                "region": series["region"],
                "year": ENVIRONMENT_DEFAULT_YEAR,
                "value": series[column],
            }
        )
        frames.append(
            IndicatorFrame(
                workbook=workbook.name,
                sheet=sheet_name,
                indicator_id=indicator_id,
                indicator_name=meta["name"],
                unit=meta["unit"],
                frame=tidy,
            )
        )
    return frames


def _load_waste_indicators(workbook: Path) -> List[IndicatorFrame]:
    if workbook.name != STATE_WASTE_WORKBOOK_NAME:
        return []
    sheet_name = "Raw Data Table"
    try:
        df = pd.read_excel(workbook, sheet_name=sheet_name, header=1)
    except ValueError:
        LOGGER.warning("Worksheet %s missing in %s", sheet_name, workbook.name)
        return []
    if "State/Territory" not in df.columns:
        LOGGER.warning(
            "Skip %s - %s missing 'State/Territory' column", workbook.name, sheet_name
        )
        return []
    df = df.copy()
    df = df[df["State/Territory"].notna()].copy()
    df["state_name"] = df["State/Territory"].astype(str).str.strip()
    df = df[df["state_name"] != ""]
    df["region"] = df["state_name"].apply(_state_name_to_code)
    missing_states = sorted(df.loc[df["region"].isna(), "state_name"].dropna().unique())
    if missing_states:
        LOGGER.warning(
            "Waste data rows skipped due to unknown states: %s", ", ".join(missing_states)
        )
    df = df[df["region"].notna()].copy()
    if df.empty:
        LOGGER.warning("All waste rows skipped after state normalization in %s", workbook.name)
        return []
    if "Data year" in df.columns:
        df["source_year"] = pd.to_numeric(df["Data year"], errors="coerce").astype("Int64")
    else:
        df["source_year"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["year"] = ENVIRONMENT_DEFAULT_YEAR

    frames: List[IndicatorFrame] = []
    for column, meta in STATE_WASTE_COLUMN_METADATA.items():
        if column not in df.columns:
            LOGGER.warning(
                "Skip %s - %s missing column '%s'", workbook.name, sheet_name, column
            )
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        mask = ~numeric.isna()
        if not mask.any():
            continue
        indicator_id = slugify(f"{workbook.stem}-{sheet_name}-{column}")
        tidy = df.loc[mask, ["region", "year", "source_year"]].copy()
        tidy.insert(0, "indicator_id", indicator_id)
        tidy["value"] = numeric.loc[mask].astype(float).values
        frames.append(
            IndicatorFrame(
                workbook=workbook.name,
                sheet=sheet_name,
                indicator_id=indicator_id,
                indicator_name=meta["name"],
                unit=meta["unit"],
                frame=tidy.loc[:, ["indicator_id", "region", "year", "value", "source_year"]],
            )
        )
    return frames


def _load_sports_factors_indicators(data_dir: Path) -> List[IndicatorFrame]:
    """Load sports factors indicators from CSV files."""
    frames: List[IndicatorFrame] = []

    # Get NFL states list first
    nfl_states = set()
    capacity_file = data_dir / "nfl_stadium_capacity.csv"
    if capacity_file.exists():
        df_capacity = pd.read_csv(capacity_file)
        nfl_states = set(df_capacity["State"].str.upper().unique())

    # Load NFL stadium capacity data
    capacity_file = data_dir / "nfl_stadium_capacity.csv"
    if capacity_file.exists():
        try:
            df_capacity = pd.read_csv(capacity_file)
            df_capacity["State"] = df_capacity["State"].str.upper()

            # Convert state names to codes
            df_capacity["region"] = df_capacity["State"].apply(_state_name_to_code)

            # Remove rows where state code couldn't be determined
            df_capacity = df_capacity[df_capacity["region"].notna()].copy()

            # Group by state and calculate average capacity
            state_capacity = df_capacity.groupby("region")["Capacity"].mean().reset_index()
            state_capacity = state_capacity.rename(columns={"Capacity": "value"})

            # Create indicator for NFL stadium capacity
            indicator_id = "sports-factors-nfl-stadium-capacity"
            tidy_capacity = pd.DataFrame({
                "indicator_id": indicator_id,
                "region": state_capacity["region"],
                "year": ENVIRONMENT_DEFAULT_YEAR,
                "value": state_capacity["value"],
            })

            frames.append(IndicatorFrame(
                workbook="nfl_stadium_capacity.csv",
                sheet="data",
                indicator_id=indicator_id,
                indicator_name=SPORTS_FACTORS_METADATA["nfl_stadium_capacity"]["name"],
                unit=SPORTS_FACTORS_METADATA["nfl_stadium_capacity"]["unit"],
                frame=tidy_capacity,
            ))
        except Exception as exc:
            LOGGER.warning("Failed to load NFL stadium capacity data: %s", exc)

    # Load NFL stadium hotels distance data
    hotels_file = data_dir / "nfl_stadium_hotels_from_gmaps.csv"
    if hotels_file.exists():
        try:
            df_hotels = pd.read_csv(hotels_file)
            df_hotels["state"] = df_hotels["state"].str.upper()

            # Convert state names to codes
            df_hotels["region"] = df_hotels["state"].apply(_state_name_to_code)

            # Remove rows where state code couldn't be determined
            df_hotels = df_hotels[df_hotels["region"].notna()].copy()

            # Convert distance from string to numeric (remove 'm' suffix if present)
            df_hotels["distance_m"] = pd.to_numeric(
                df_hotels["distance_m"].astype(str).str.replace(',', '').str.extract(r'(\d+\.?\d*)')[0],
                errors="coerce"
            )

            # Group by state and calculate average distance
            state_distance = df_hotels.groupby("region")["distance_m"].mean().reset_index()
            state_distance = state_distance.rename(columns={"distance_m": "value"})

            # Create indicator for stadium to hotel distance
            indicator_id = "sports-factors-stadium-hotel-distance-avg"
            tidy_distance = pd.DataFrame({
                "indicator_id": indicator_id,
                "region": state_distance["region"],
                "year": ENVIRONMENT_DEFAULT_YEAR,
                "value": state_distance["value"],
            })

            frames.append(IndicatorFrame(
                workbook="nfl_stadium_hotels_from_gmaps.csv",
                sheet="data",
                indicator_id=indicator_id,
                indicator_name=SPORTS_FACTORS_METADATA["stadium_hotel_distance_avg"]["name"],
                unit=SPORTS_FACTORS_METADATA["stadium_hotel_distance_avg"]["unit"],
                frame=tidy_distance,
            ))
        except Exception as exc:
            LOGGER.warning("Failed to load NFL stadium hotels distance data: %s", exc)

    # Load peak public transport capacity data
    transport_file = data_dir / "peak_capacity_by_state_2023.csv"
    if transport_file.exists():
        try:
            df_transport = pd.read_csv(transport_file)
            df_transport["State"] = df_transport["State"].str.upper()

            # Create indicator for peak public transport capacity
            indicator_id = "sports-factors-peak-public-transport-capacity"
            tidy_transport = pd.DataFrame({
                "indicator_id": indicator_id,
                "region": df_transport["State"],
                "year": ENVIRONMENT_DEFAULT_YEAR,
                "value": df_transport["Peak_Capacity"],
            })

            frames.append(IndicatorFrame(
                workbook="peak_capacity_by_state_2023.csv",
                sheet="data",
                indicator_id=indicator_id,
                indicator_name=SPORTS_FACTORS_METADATA["peak_public_transport_capacity"]["name"],
                unit=SPORTS_FACTORS_METADATA["peak_public_transport_capacity"]["unit"],
                frame=tidy_transport,
            ))
        except Exception as exc:
            LOGGER.warning("Failed to load peak public transport capacity data: %s", exc)

    # Load and process transport time data
    transport_file = data_dir / "state_transport_stats.csv"
    if transport_file.exists() and nfl_states:
        try:
            df_transport = pd.read_csv(transport_file)

            # Filter to only NFL states
            df_transport = df_transport[
                (df_transport["State_A"].str.upper().isin(nfl_states)) &
                (df_transport["State_B"].str.upper().isin(nfl_states))
            ].copy()

            # Calculate average times for each state
            avg_times = []

            for state in nfl_states:
                # Driving time: average time between this state and other NFL states (bidirectional)
                driving_times_a_to_b = df_transport[
                    (df_transport["State_A"].str.upper() == state) &
                    (df_transport["State_B"].str.upper() != state) &
                    (df_transport["State_B"].str.upper().isin(nfl_states))
                ]["Driving_Time_s"].dropna()

                driving_times_b_to_a = df_transport[
                    (df_transport["State_B"].str.upper() == state) &
                    (df_transport["State_A"].str.upper() != state) &
                    (df_transport["State_A"].str.upper().isin(nfl_states))
                ]["Driving_Time_s"].dropna()

                all_driving_times = pd.concat([driving_times_a_to_b, driving_times_b_to_a])

                # Flight time: average time between this state and other NFL states (bidirectional)
                flight_times_a_to_b = df_transport[
                    (df_transport["State_A"].str.upper() == state) &
                    (df_transport["State_B"].str.upper() != state) &
                    (df_transport["State_B"].str.upper().isin(nfl_states))
                ]["Flight_Time_s"].dropna()

                flight_times_b_to_a = df_transport[
                    (df_transport["State_B"].str.upper() == state) &
                    (df_transport["State_A"].str.upper() != state) &
                    (df_transport["State_A"].str.upper().isin(nfl_states))
                ]["Flight_Time_s"].dropna()

                all_flight_times = pd.concat([flight_times_a_to_b, flight_times_b_to_a])

                avg_driving = all_driving_times.mean() if len(all_driving_times) > 0 else None
                avg_flight = all_flight_times.mean() if len(all_flight_times) > 0 else None

                if avg_driving is not None or avg_flight is not None:
                    # Convert state name back to code format for consistency
                    state_code = _state_name_to_code(state)
                    if state_code:
                        avg_times.append({
                            "region": state_code,
                            "avg_driving_time": avg_driving,
                            "avg_flight_time": avg_flight
                        })

            # Create driving time indicator
            if any(item["avg_driving_time"] is not None for item in avg_times):
                driving_data = [
                    {"indicator_id": "sports-factors-avg-driving-time-to-nfl-states", "region": item["region"], "year": ENVIRONMENT_DEFAULT_YEAR, "value": item["avg_driving_time"]}
                    for item in avg_times if item["avg_driving_time"] is not None
                ]
                if driving_data:
                    df_driving = pd.DataFrame(driving_data)
                    frames.append(IndicatorFrame(
                        workbook="state_transport_stats.csv",
                        sheet="driving_times",
                        indicator_id="sports-factors-avg-driving-time-to-nfl-states",
                        indicator_name=SPORTS_FACTORS_METADATA["avg_driving_time_to_nfl_states"]["name"],
                        unit=SPORTS_FACTORS_METADATA["avg_driving_time_to_nfl_states"]["unit"],
                        frame=df_driving,
                    ))

            # Create flight time indicator
            if any(item["avg_flight_time"] is not None for item in avg_times):
                flight_data = [
                    {"indicator_id": "sports-factors-avg-flight-time-to-nfl-states", "region": item["region"], "year": ENVIRONMENT_DEFAULT_YEAR, "value": item["avg_flight_time"]}
                    for item in avg_times if item["avg_flight_time"] is not None
                ]
                if flight_data:
                    df_flight = pd.DataFrame(flight_data)
                    frames.append(IndicatorFrame(
                        workbook="state_transport_stats.csv",
                        sheet="flight_times",
                        indicator_id="sports-factors-avg-flight-time-to-nfl-states",
                        indicator_name=SPORTS_FACTORS_METADATA["avg_flight_time_to_nfl_states"]["name"],
                        unit=SPORTS_FACTORS_METADATA["avg_flight_time_to_nfl_states"]["unit"],
                        frame=df_flight,
                    ))

        except Exception as exc:
            LOGGER.warning("Failed to load transport time data: %s", exc)

    return frames


def _load_environment_indicator_frames(workbook: Path) -> List[IndicatorFrame]:
    if not workbook.exists():
        return []
    frames: List[IndicatorFrame] = []
    frames.extend(_load_city_water_indicators(workbook))
    frames.extend(_load_air_travel_indicators(workbook))
    frames.extend(_load_waste_indicators(workbook))
    return frames


def load_all_indicators(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all Excel files in data_dir and return (long data, metadata)."""

    records: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, object]] = []

    env_workbook = data_dir / ENVIRONMENT_WORKBOOK_NAME
    waste_workbook = data_dir / STATE_WASTE_WORKBOOK_NAME
    for workbook in sorted(data_dir.glob("*.xlsx")):
        if workbook.name.startswith("~$") or workbook.name.startswith(".~"):
            LOGGER.debug("Skip temporary workbook %s", workbook.name)
            continue
        if workbook.name in {ENVIRONMENT_WORKBOOK_NAME, STATE_WASTE_WORKBOOK_NAME}:
            continue
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
    if env_workbook.exists():
        LOGGER.info("Loading supplemental environment indicators from %s", env_workbook.name)
        env_frames = _load_environment_indicator_frames(env_workbook)
        for indicator in env_frames:
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
    if waste_workbook.exists():
        LOGGER.info("Loading supplemental waste indicators from %s", waste_workbook.name)
        waste_frames = _load_waste_indicators(waste_workbook)
        for indicator in waste_frames:
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

    # Load sports factors indicators
    LOGGER.info("Loading sports factors indicators from CSV files")
    sports_frames = _load_sports_factors_indicators(data_dir)
    for indicator in sports_frames:
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
