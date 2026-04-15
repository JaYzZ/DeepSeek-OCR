#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from Qwen.evaluation.m3cot.dataset_utils import deterministic_limit


def test_m3cot_deterministic_limit_matches_domain_distribution_on_clustered_input():
    rows = []
    for idx in range(455):
        rows.append({"id": f"commonsense-{idx}", "index": f"commonsense-{idx}", "domain": "commonsense"})
    for idx in range(1622):
        rows.append({"id": f"science-{idx}", "index": f"science-{idx}", "domain": "science"})
    for idx in range(241):
        rows.append({"id": f"mathematics-{idx}", "index": f"mathematics-{idx}", "domain": "mathematics"})

    data = pd.DataFrame(rows)
    limited = deterministic_limit(data, 100)

    counts = limited["domain"].value_counts().to_dict()
    assert len(limited) == 100
    assert counts == {
        "science": 70,
        "commonsense": 20,
        "mathematics": 10,
    }
