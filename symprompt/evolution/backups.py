from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def backup_database(db_path: Path) -> Path | None:
    """
    Create a timestamped backup of the OpenEvolve database directory.

    Mirrors the approach used in the evolve project: copies the entire
    evolution.db directory to a sibling backup directory with a timestamp.
    """
    if not db_path.exists():
        return None

    parent = db_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"evolution_backup_{timestamp}.db"
    backup_path = parent / backup_name

    shutil.copytree(db_path, backup_path)
    return backup_path

