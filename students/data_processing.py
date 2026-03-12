"""
Data loading and preprocessing functions for the heart disease dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_TARGET_ALIASES = {
    "cholesterol": "chol",
    "heart_disease": "num",
    "target": "num",
}
