"""
pipeline.py — Pipeline de données silencieux, exécutable en thread de fond.
Aucun print. Toute la communication passe par `state`.
"""
import io
import logging
import contextlib
import threading
from datetime import datetime

import build_master_db
import get_race_results
from fusion_data import create_hybrid_dataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# État partagé thread-safe (écritures atomiques sur types simples)
# ─────────────────────────────────────────────────────────────
state = {
    "running":   False,
    "done":      False,
    "progress":  0.0,
    "step":      "",
    "error":     None,
}

_lock = threading.Lock()


def _set(**kwargs):
    with _lock:
        state.update(kwargs)


def _silent(func, *args, **kwargs):
    """Exécute une fonction en supprimant tout stdout/stderr."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        func(*args, **kwargs)


def _patch_years():
    """Cible toujours year-1 et year courant — jamais hardcodé."""
    year = datetime.now().year
    build_master_db.YEARS_TO_UPDATE   = [year - 1, year]
    get_race_results.YEARS_TO_UPDATE  = [year - 1, year]


# ─────────────────────────────────────────────────────────────
# ÉTAPES DU PIPELINE
# ─────────────────────────────────────────────────────────────
_STEPS = [
    (0.00, "Initialisation..."),
    (0.05, "Synchronisation des temps au tour (OpenF1)..."),
    (0.55, "Synchronisation des classements (OpenF1)..."),
    (0.80, "Fusion et calcul des rythmes..."),
    (1.00, "✅ Base de données à jour"),
]


def run():
    """
    Lance le pipeline complet.
    À appeler dans un thread de fond : threading.Thread(target=pipeline.run).
    """
    _set(running=True, done=False, error=None, progress=0.0, step="Démarrage...")

    try:
        _set(progress=_STEPS[0][0], step=_STEPS[0][1])
        _patch_years()

        _set(progress=_STEPS[1][0], step=_STEPS[1][1])
        _silent(build_master_db.update_master_database)

        _set(progress=_STEPS[2][0], step=_STEPS[2][1])
        _silent(get_race_results.build_results_db)

        _set(progress=_STEPS[3][0], step=_STEPS[3][1])
        _silent(create_hybrid_dataset)

        _set(progress=_STEPS[4][0], step=_STEPS[4][1])

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        _set(error=str(e), step=f"❌ Erreur : {e}")

    finally:
        _set(running=False, done=True)


def start_in_background() -> threading.Thread:
    """Lance le pipeline dans un daemon thread et retourne la référence."""
    t = threading.Thread(target=run, daemon=True, name="f1-pipeline")
    t.start()
    return t
