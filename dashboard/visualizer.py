import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd


# Utility Formatters
def format_lap_time(seconds):
    """Format lap time in MM:SS.mmm format for human-readable tooltips."""
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02}:{sec:02}.{millis:03}"


def format_seconds_to_mmss(seconds):
    """Format seconds into MM:SS string for Y-axis tick labels."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02}:{secs:02}"


# Lap Time Chart
def plot_lap_times(lap_time_df: pd.DataFrame, color_map: dict):
    """
    Create a line chart showing lap times per driver over the race distance.

    Input data comes from OpenF1's /laps endpoint, processed and filtered.
    Pit exit laps (e.g. out-laps) are flagged and marked in tooltips.

    Args:
        lap_time_df (pd.DataFrame): Cleaned lap data.
        color_map (dict): Driver acronym to team color.

    Returns:
        Plotly Figure object
    """
    if lap_time_df.empty:
        st.warning("No lap data available for this session.")
        return None

    lap_time_df["formatted_lap_time"] = lap_time_df["lap_duration"].apply(format_lap_time)
    lap_time_df["is_pit_out_lap"] = lap_time_df["is_pit_out_lap"].fillna(False).astype(bool)

    fig = go.Figure()

    for driver in lap_time_df["name_acronym"].unique():
        driver_data = lap_time_df[lap_time_df["name_acronym"] == driver].copy()
        driver_data = driver_data.sort_values("lap_number")

        # Custom tooltip for each data point
        hover_texts = [
            f"<b>{driver}: {row['driver_number']}</b><br>"
            f"Lap: {row['lap_number']}<br>"
            f"Lap Time: {row['formatted_lap_time']}"
            + ("<br>🔧 PIT" if row['is_pit_out_lap'] else "")
            for _, row in driver_data.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=driver_data["lap_number"],
            y=driver_data["lap_duration"],
            mode="lines+markers",
            name=driver,
            marker=dict(color=color_map.get(driver, "gray")),
            line=dict(color=color_map.get(driver, "gray")),
            hoverinfo="text",
            hovertext=hover_texts,
        ))

    fig.update_layout(
        title="Lap Times by Driver",
        xaxis_title="Lap",
        yaxis_title="Lap Time (MM:SS)",
        hovermode="closest",
        height=600,
    )

    # Format Y-axis to readable MM:SS format
    tick_vals = sorted(lap_time_df["lap_duration"].dropna().unique())
    tick_vals = [round(val, 0) for val in tick_vals if 60 <= val <= 180]  # clean range
    tick_vals = sorted(set(tick_vals))[::5]  # fewer ticks, every ~5 sec

    fig.update_yaxes(
        tickvals=tick_vals,
        ticktext=[format_seconds_to_mmss(val) for val in tick_vals],
    )

    return fig


# Tire Strategy Chart
# Map Pirelli compounds to colors matches standard F1 graphics
COMPOUND_COLORS = {
    "SOFT": "red",
    "MEDIUM": "yellow",
    "HARD": "white",
    "INTERMEDIATE": "green",
    "WET": "blue",
    "Unknown": "gray"
}


def plot_tire_strategy(stints_df, color_map: dict):
    """
    Show tire compound strategy for each driver using horizontal bars.

    Uses OpenF1 /stints endpoint to show start/end lap and compound used.

    Args:
        stints_df (pd.DataFrame): Cleaned tire stint data.
        color_map (dict): Driver acronym to team color.

    Returns:
        Plotly Figure object
    """
    if stints_df.empty:
        st.warning("No stint data available.")
        return None

    fig = go.Figure()

    for _, row in stints_df.iterrows():
        compound = row["compound"].upper()
        acronym = row["name_acronym"]

        fig.add_trace(go.Bar(
            x=[row["lap_count"]],  # Width of bar = number of laps
            y=[acronym],  # One row per driver
            base=row["lap_start"],  # Start lap (bar offset)
            orientation="h",
            marker=dict(color=COMPOUND_COLORS.get(compound, "gray")),
            hovertemplate=(
                f"{acronym}: {row['driver_number']}<br>"
                f"Compound: {compound}<br>"
                f"Laps: {row['lap_count']}<br>"
                f"Start Lap: {row['lap_start']}<br>"
                f"End Lap: {row['lap_end']}"
            ),
            name="",
            showlegend=False
        ))

        # Add colored annotations instead of y-ticks
    y_labels = stints_df["name_acronym"].unique()
    for acronym in y_labels:
        fig.add_annotation(
            x=-3,  # offset left
            y=acronym,
            xref="x",
            yref="y",
            text=f"<b>{acronym}</b>",
            showarrow=False,
            font=dict(
                color=color_map.get(acronym, "#AAA"),  # driver color from map
                size=12
            ),
            align="right"
        )

    fig.update_layout(
        title="Tire Strategy by Driver",
        xaxis_title="Lap Number",
        yaxis_title="",
        barmode="stack",
        height=600,
        margin=dict(l=120),  # make room for left-side labels
    )

    # Hide original Y ticks
    fig.update_yaxes(showticklabels=False)

    return fig


# Pit Stop Duration Chart
def plot_pit_stop(pit_stop_df: pd.DataFrame, color_map: dict):
    """
    Compare pit stop durations across drivers.

    Data comes from OpenF1 /pit endpoint, with pit_duration per lap.

    Args:
        pit_stop_df (pd.DataFrame): Cleaned pit stop data.
        color_map (dict): Driver acronym to team color.

    Returns:
        Plotly Figure object
    """
    if pit_stop_df.empty:
        st.warning("No pit stop data available for this session.")
        return None

    pit_stop_df["driver_number"] = pit_stop_df["driver_number"].astype(str)

    # Combine acronym + number in one column for labeling
    pit_stop_df["driver_label"] = pit_stop_df["name_acronym"] + ": " + pit_stop_df["driver_number"]

    fig = px.bar(
        pit_stop_df,
        x="lap_number",
        y="pit_duration",
        color="name_acronym",
        color_discrete_map=color_map,
        hover_data={
            "driver_label": False,
            "lap_number": False,  # We'll handle this in custom_data
            "pit_duration": False,  # We'll handle this in custom_data
            "name_acronym": False,  # We'll handle this in custom_data
            "driver_number": False,  # We'll handle this in custom_data
        },
        custom_data=["name_acronym", "driver_number", "lap_number", "pit_duration"],
        labels={
            "lap_number": "Lap",
            "pit_duration": "Time in pit lane (s)",
        }
    )

    # Customize the hover template
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}: %{customdata[1]}</b><br>" +
                      "Lap: %{customdata[2]}<br>" +
                      "Time in pit lane (s): %{customdata[3]:.1f}<br>" +
                      "<extra></extra>"  # Removes the trace box
    )
    fig.update_layout(
        title="Pit Stop Times by Driver",
        hovermode="closest",
        barmode="group",
        height=600)
    return fig
