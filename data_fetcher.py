"""
data_fetcher.py — Source unique de vérité pour toutes les données dynamiques.
Tout vient de l'API OpenF1. Rien n'est hardcodé.
"""
import requests
import logging
from datetime import datetime, timezone
from functools import lru_cache
from sessions import get_session_list

logger = logging.getLogger(__name__)
BASE_URL = "https://api.openf1.org/v1"


def _get(endpoint: str, params: dict, timeout: int = 10) -> list:
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"API {endpoint} failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# CALENDRIER
# ─────────────────────────────────────────────────────────────

def get_season_races(year: int | None = None) -> list[dict]:
    """Toutes les courses d'une saison, triées par date."""
    year = year or datetime.now().year
    sessions = get_session_list(year=year, session_name="Race")
    return sorted(sessions, key=lambda s: s.get("date_start", ""))


def get_next_race(year: int | None = None) -> dict | None:
    """Prochaine course à venir selon l'API."""
    now = datetime.now(timezone.utc)
    races = get_season_races(year)

    for race in races:
        date_str = race.get("date_start", "")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > now:
                return race
        except Exception:
            continue
    return None


def get_last_completed_race(year: int | None = None) -> dict | None:
    """Dernière course terminée de la saison."""
    now = datetime.now(timezone.utc)
    races = get_season_races(year)
    last = None
    for race in races:
        date_str = race.get("date_start", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < now:
                last = race
        except Exception:
            continue
    return last


def get_circuit_list(year: int | None = None) -> list[str]:
    """Liste des circuits de la saison depuis l'API."""
    races = get_season_races(year or datetime.now().year)
    seen = dict.fromkeys(r["location"] for r in races if r.get("location"))
    return list(seen.keys())  # Ordre calendrier préservé


# ─────────────────────────────────────────────────────────────
# GRILLE DE PILOTES
# ─────────────────────────────────────────────────────────────

def get_current_grid(year: int | None = None) -> list[tuple[int, str]]:
    """
    Grille de pilotes (numéro, équipe) depuis la dernière course disponible.
    Fallback sur l'année précédente si la saison n'a pas encore démarré.
    """
    year = year or datetime.now().year

    last = get_last_completed_race(year)
    if not last:
        last = get_last_completed_race(year - 1)
    if not last:
        logger.error("Aucune session disponible pour construire la grille.")
        return []

    session_key = last["session_key"]
    drivers_data = _get("drivers", {"session_key": session_key})

    # Dédoublonnage : un pilote peut apparaître plusieurs fois
    seen: dict[int, str] = {}
    for d in drivers_data:
        num = d.get("driver_number")
        team = d.get("team_name", "Unknown")
        if num and num not in seen:
            seen[num] = team

    return list(seen.items())  # [(num, team), ...]


# ─────────────────────────────────────────────────────────────
# ACRONYMES PILOTES (avec cache local pour éviter les appels répétés)
# ─────────────────────────────────────────────────────────────

_acronym_cache: dict[int, str] = {}

def get_acronym(driver_number: int, session_key: int | None = None) -> str:
    """Acronyme d'un pilote (ex: 1 → VER). Mis en cache."""
    if driver_number in _acronym_cache:
        return _acronym_cache[driver_number]

    params = {"driver_number": driver_number}
    if session_key:
        params["session_key"] = session_key

    data = _get("drivers", params, timeout=5)
    if data:
        acr = data[0].get("name_acronym", "???")
        _acronym_cache[driver_number] = acr
        return acr

    fallback = f"#{driver_number}"
    _acronym_cache[driver_number] = fallback
    return fallback


def preload_acronyms(grid: list[tuple[int, str]], session_key: int | None = None):
    """Pré-charge tous les acronymes en une seule passe."""
    for d_num, _ in grid:
        if d_num not in _acronym_cache:
            get_acronym(d_num, session_key)
