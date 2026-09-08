"""Recompute the complete family grid using rational arithmetic."""
import math
from exact_green import certify_exact, energy_nullity, endpoint_structure
from revision_claims import expected_grid


def recheck(data):
    rows = data["rows"]
    assert len(rows) == 152
    assert {tuple(row["key"]) for row in rows} == set(expected_grid())
    certified_count = 0
    for index, row in enumerate(rows):
        d, k, l, n, eta = row["key"]
        energy = energy_nullity(d, k, l, n, eta)
        continuity = d - 1 if eta is None else eta
        endpoint = endpoint_structure(d, k, l, n, continuity) if energy["definite"] else None
        for name, degree in [("base", 96), ("recheck", 192)]:
            stored = row[name]
            assert stored["D"] == degree
            assert stored["K_definite"] == energy["definite"]
            assert stored["K_nullity"] == energy["nullity"]
            assert stored["r"] == energy["r"]
            if energy["definite"]:
                exact = certify_exact(d, k, l, n, continuity, D=degree)
                verdict = exact["strict"] or (exact["nonneg"] and endpoint["closed_square_ok"])
                assert stored["strict"] == exact["strict"]
                assert stored["nonneg"] == exact["nonneg"]
                assert stored["closed_square_ok"] == endpoint["closed_square_ok"]
                assert stored["n_live_endpoints"] == endpoint["n_live_endpoints"]
                assert math.isclose(stored["elev_min_rel"], float(exact["elev_min_rel"]),
                                    rel_tol=1e-13, abs_tol=1e-18)
            else:
                verdict = False
                assert not stored["strict"] and not stored["nonneg"]
                assert not stored["closed_square_ok"]
            assert stored["certified"] == verdict
            if name == "base":
                certified_count += int(verdict)
        if (index + 1) % 20 == 0:
            print("Rational family replay", index + 1, "/", len(rows), flush=True)
    assert certified_count == data["n_pass"]
    return {"families": len(rows), "elevation_degrees": [96, 192],
            "evaluations": 2 * len(rows), "certified_families": certified_count}
