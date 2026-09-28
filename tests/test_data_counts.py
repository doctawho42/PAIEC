"""experiments/data_counts.py on synthetic tables (no data needed)."""
import pandas as pd

from experiments import data_counts as DC


def test_features_parses_item_features():
    assert DC.features("competition=aime_2025;problem_idx=2") == {"competition": "aime_2025",
                                                                   "problem_idx": "2"}
    assert DC.features("") == {} and DC.features(None) == {}


def _resp(rows):
    return pd.DataFrame(rows, columns=["subject_id", "item_id", "benchmark_id", "response"])


def test_pair_and_matharena_counts():
    rows = []
    for i in range(80):                          # s1: 80 distinct items, eligible
        rows.append(("s1", f"i{i}", "b", float(i % 2)))
    rows.append(("s1", "i0", "b", 0.5))          # a non-binary response is dropped
    for i in range(79):                          # s2: 79 items, below the floor
        rows.append(("s2", f"i{i}", "b", 1.0))
    rows.append(("s2", "j0", "b", 1.0))          # ...but 80 distinct: eligible
    rows.append(("s3", "k0", "b", 0.0))          # s3: one item, below the floor
    resp = _resp(rows)
    c = DC.pair_counts(resp)
    assert c["binary_responses"] == len(rows) - 1
    assert c["eligible_pairs"] == 2 and c["responses_in_eligible_pairs"] == 160
    assert c["distinct_items_in_eligible_pairs"] == 81
    items = pd.DataFrame({"item_id": [f"i{i}" for i in range(80)] + ["j0", "k0", "unused"],
                          "item_features": ["competition=kangaroo_2025_1_2"] * 2
                          + ["competition=aime_2025"] * 78 + ["competition=hmmt", "competition=imc_2025",
                                                              "competition=x"]})
    m = DC.matharena_counts(resp, items)
    assert m["items_with_a_binary_response"] == 82 and m["competitions_of_those_items"] == 4
    assert m["items_in_eligible_pairs"] == 81 and m["competitions_in_eligible_pairs"] == 3
    assert m["competitions_only_below_the_floor"] == ["imc_2025"]
    assert m["kangaroo_items"]["in_eligible_pairs"] == 2
