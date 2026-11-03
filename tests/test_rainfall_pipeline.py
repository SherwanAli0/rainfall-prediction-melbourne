"""Offline checks for rainfall_prediction.py using a tiny synthetic dataset.

The script is executed with pandas.read_csv patched, so nothing is downloaded.
The script is copied to a temporary directory first so that the figures it saves
do not overwrite the PNG files committed in images/.
"""
import runpy
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "rainfall_prediction.py"


def make_fixture(n_per_location=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for location in ["Melbourne", "MelbourneAirport", "Watsonia", "Sydney"]:
        n = n_per_location
        humidity3pm = rng.uniform(20, 95, n)
        rainfall = np.where(rng.uniform(size=n) < 0.25, rng.uniform(1, 20, n), 0.0)
        frames.append(pd.DataFrame({
            "Date": pd.date_range("2015-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "Location": location,
            "MinTemp": rng.normal(10, 4, n),
            "MaxTemp": rng.normal(20, 5, n),
            "Rainfall": rainfall,
            "WindGustDir": rng.choice(["N", "S", "E", "W"], n),
            "WindGustSpeed": rng.uniform(10, 80, n),
            "Humidity3pm": humidity3pm,
            "Pressure3pm": rng.normal(1013, 6, n),
            "RainToday": np.where(rainfall >= 1, "Yes", "No"),
            "RainTomorrow": np.where(humidity3pm + rng.normal(0, 10, n) > 70, "Yes", "No"),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    work = tmp_path_factory.mktemp("rainfall")
    script = work / "rainfall_prediction.py"
    shutil.copy(SCRIPT, script)
    fixture = make_fixture()
    patch = pytest.MonkeyPatch()
    patch.setattr(pd, "read_csv", lambda *args, **kwargs: fixture.copy())
    try:
        namespace = runpy.run_path(str(script), run_name="__main__")
    finally:
        patch.undo()
    return namespace, fixture, work


def test_labels_are_renamed_as_documented(run):
    ns, fixture, _ = run
    df = ns["df"]
    original = fixture.loc[df.index]
    assert set(df["Location"]) <= {"Melbourne", "MelbourneAirport", "Watsonia"}
    assert (df["RainYesterday"] == original["RainToday"]).all()
    assert (df["RainToday"] == original["RainTomorrow"]).all()
    assert "RainTomorrow" not in df.columns


def test_target_is_not_a_feature(run):
    ns, _, _ = run
    assert "RainToday" not in ns["X"].columns
    assert "RainToday" not in ns["numeric_features"] + ns["categorical_features"]
    assert ns["y"].name == "RainToday"


def test_feature_names_match_importances(run):
    ns, _, _ = run
    names = ns["feature_names"]
    assert len(names) == len(ns["feature_importances"])
    assert names[: len(ns["numeric_features"])] == ns["numeric_features"]
    assert "RainYesterday_Yes" in names
    assert not any(n.startswith("RainToday") for n in names)


def test_cv_is_five_fold(run):
    ns, _, _ = run
    assert ns["grid_search"].cv == 5


def test_documented_figures_are_saved(run):
    _, _, work = run
    for name in ("confusion_matrix.png", "feature_importance.png"):
        path = work / "images" / name
        assert path.is_file() and path.stat().st_size > 0
