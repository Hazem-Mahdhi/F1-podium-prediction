import pandas as pd


def process_lap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare lap data for visualization.

    - Filters out laps without duration.
    - Sorts by driver number and lap number.

    Args:
        df (pd.DataFrame): Raw lap data from API.

    Returns:
        pd.DataFrame: Cleaned and sorted lap data.
    """
    if df.empty:
        return df

    df = df[df['lap_duration'].notna()]  # Drop laps missing duration info (i.e. retirements or red flags)
    df = df.sort_values(['driver_number', 'lap_number'])  # Sort for logical order in lap-time visualization
    return df


def process_stints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare stint data for the tire strategy chart.

    - Sorts by driver and stint number.
    - Fills missing compound values with "Unknown".
    - Adds a lap_count column.

    Args:
        df (pd.DataFrame): Raw stint data.

    Returns:
        pd.DataFrame: Cleaned stint data with compound info and lap counts.
    """
    if df.empty:
        return df

    df = df.sort_values(by=["driver_number", "stint_number"])  # Sort by driver and stint sequence
    df["compound"] = df["compound"].fillna("Unknown")  # Replace missing compound with placeholder
    df["lap_count"] = df["lap_end"] - df["lap_start"] + 1  # Compute total laps in each stint
    return df


def process_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare pit stop data for charting.

    - Filters out entries without a recorded duration.
    - Sorts by driver and lap number.

    Args:
        df (pd.DataFrame): Raw pit stop data.

    Returns:
        pd.DataFrame: Cleaned and sorted pit stop data.
    """
    if df.empty:
        return df

    df = df[df["pit_duration"].notna()]  # Only keep pit stops with a recorded duration
    df = df.sort_values(by=["driver_number", "lap_number"])  # Organize by race sequence
    return df


def build_driver_color_map(driver_df: pd.DataFrame) -> dict:
    """
    Build a dictionary that maps driver acronyms to their team color.

    Args:
        driver_df (pd.DataFrame): DataFrame with driver and team information.

    Returns:
        dict: Dictionary mapping name_acronym to team_colour.
    """
    if driver_df.empty:
        return {}

    # Format team colors to always start with '#' for valid CSS color input
    driver_df["team_colour"] = driver_df["team_colour"].apply(
        lambda x: f"#{x}" if not str(x).startswith("#") else x
    )
    # Plotly prefers string keys; ensure driver_number is string
    driver_df["driver_number"] = driver_df["driver_number"].astype(str)

    # Build the mapping from acronym to team color
    color_map = {
        str(row["name_acronym"]): row["team_colour"]
        for _, row in driver_df.iterrows()
        if pd.notna(row["team_colour"])
    }

    return color_map
