import streamlit as st
import requests
import pandas as pd
BASE_URL = "https://api.openf1.org/v1/"
import os


def fetch_data(endpoint, params=None):
    """
    Fetch data from the OpenF1 API and return it as a DataFrame.

    Args:
        endpoint (str): API endpoint (e.g., "meetings", "sessions").
        params (dict): Optional query parameters for the API.

    Returns:
        pd.DataFrame: DataFrame containing the API response data.

    Notes:
        The OpenF1 API requires properly URL-encoded query strings,
        especially when using complex filters (e.g., strings with spaces).
        Using `requests.get(url, params=params)` sometimes causes issues with
        formatting, so we manually prepare the full URL using `requests.Request`.
    """
    if params is None:
        params = {}

    url = f"{BASE_URL}{endpoint}"
    full_url = requests.Request('GET', url, params=params).prepare().url
    response = requests.get(full_url)
    response.raise_for_status()
    return pd.DataFrame(response.json())


# Cached API calls using Streamlit's cache_data decorator

@st.cache_data
def fetch_meetings(year, country):
    # The 'meetings' endpoint returns all information for a specified meeting (Miami, Monaco, Imola, etc)
    # for a specified year (2023, 2024, etc).
    df = fetch_data("meetings", {"year": year, "country_name": country})
    if df.empty:
        st.error("⚠️ No meeting data found.")
        return pd.DataFrame()

    # Create a label for easier dropdown display
    df["label"] = df["meeting_name"] + " - " + df["location"]
    df = df.sort_values(by="meeting_key", ascending=False)

    # Return minimal relevant fields
    return df[["meeting_key", "label", "year"]].drop_duplicates()


@st.cache_data
def fetch_sessions(meeting_key):
    # The 'sessions' endpoint returns all session types (FP1, Qualifying, Race) for a specific Grand Prix.
    # Filtered here using 'meeting_key' from the 'meetings' endpoint.
    df = fetch_data("sessions", {"meeting_key": meeting_key})

    # Combine session name and start date for display
    df["label"] = df["session_name"] + " (" + df["date_start"] + ")"

    # Only keep necessary columns for dropdowns
    return df[["session_key", "label"]].drop_duplicates()


@st.cache_data
def fetch_laps(session_key):
    # Retrieves detailed lap timing data for a given session
    return fetch_data("laps", {"session_key": session_key})


@st.cache_data
def fetch_stints(session_key):
    # Fetches tire stint data, which includes tire compound and start/end laps
    return fetch_data("stints", {"session_key": session_key})


@st.cache_data
def fetch_pit_stop(session_key):
    # Returns pit stop information, including duration and lap number
    return fetch_data("pit", {"session_key": session_key})


@st.cache_data
def fetch_drivers(session_key):
    # Provides driver metadata such as name, number, and team color
    return fetch_data("drivers", {"session_key": session_key})
