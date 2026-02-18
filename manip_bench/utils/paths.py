"""Centralized asset path resolution for ManipBench.

All extension code must use the functions in this module to resolve paths
to USD assets, textures, and other data files.  Direct path construction
via ``Path(__file__).parent`` is prohibited in extension code.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _PROJECT_ROOT / "assets"


def get_project_root() -> Path:
    """Return the manip-bench repository root."""
    return _PROJECT_ROOT


def get_assets_root() -> Path:
    """Return the top-level ``assets/`` directory."""
    return _ASSETS_ROOT


def get_robot_usd_path(relative_path: str) -> str:
    """Resolve a robot USD asset under ``assets/robots/``."""
    return (_ASSETS_ROOT / "robots" / relative_path).as_posix()


def get_object_usd_path(relative_path: str) -> str:
    """Resolve an object USD asset under ``assets/objects/``."""
    return (_ASSETS_ROOT / "objects" / relative_path).as_posix()


def get_scene_usd_path(relative_path: str) -> str:
    """Resolve a scene USD asset under ``assets/scenes/``."""
    return (_ASSETS_ROOT / "scenes" / relative_path).as_posix()
