from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

@dataclass
class ModelSpec:
    name: str
    estimator: object

def get_model_specs(selected: Tuple[str, ...]) -> Dict[str, ModelSpec]:
    all_models = {
        "knn": ModelSpec("knn", KNeighborsClassifier(n_neighbors=7)),
        "rf": ModelSpec("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        "dt": ModelSpec("dt", DecisionTreeClassifier(max_depth=None, random_state=42)),
        "lr": ModelSpec("lr", LogisticRegression(max_iter=2000, n_jobs=None)),
    }
    specs = {}
    for key in selected:
        if key not in all_models:
            raise ValueError(f"Unknown model key: {key} (valid: {list(all_models.keys())})")
        specs[key] = all_models[key]
    return specs
