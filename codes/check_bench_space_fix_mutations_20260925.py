#!/usr/bin/env python3
"""Mutation tests for the space-fix entries of bench/reproduce.py (2026-09-25).

Each mutation must make `bench.reproduce.main()` return non-zero; the unmutated run must return 0.
  M0  baseline (no mutation)                                  -> 0
  M1  SF.q2_rho value in the re-extracted verdicts +0.02      -> MISMATCH
  M2  Brennan re-extracted CI lower forced to -0.01           -> MISMATCH
  M3  SF.q3_rsa count 2 -> 3 in the receipt                   -> MISMATCH
  M4  delete the SUPERSEDED marker of A.q2_rho                -> guard FAIL (stale source unflagged)
  M5  delete the SUPERSEDED marker of cohort.brennan          -> guard FAIL
  M6  SUPERSEDED entry names a missing sibling (SF.q2_rho)    -> guard FAIL
No files are written; mutations are applied in memory.
"""
import contextlib
import copy
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench.reproduce as R  # noqa: E402

REAL_JSON = R._json


def run():
    sys.argv = ["reproduce"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main()
    return rc, buf.getvalue()


def with_json(mut):
    def _j(rel):
        d = copy.deepcopy(REAL_JSON(rel))
        mut(rel, d)
        return d
    return _j


def main():
    results = []
    sup0, claims0 = dict(R.SUPERSEDED), list(R.CLAIMS)

    def case(name, setup, expect_fail):
        R._json = REAL_JSON
        R.SUPERSEDED.clear(); R.SUPERSEDED.update(sup0)
        R.CLAIMS[:] = claims0
        setup()
        rc, out = run()
        ok = (rc != 0) == expect_fail
        tail = [l for l in out.splitlines() if "MISMATCH" in l or "GUARD" in l][:2]
        results.append((name, rc, ok, tail))

    case("M0 baseline", lambda: None, False)
    case("M1 SF.q2_rho +0.02", lambda: setattr(R, "_json", with_json(
        lambda rel, d: rel.endswith("adr0006_registered_verdicts_spacefix.json") and d["q2"].__setitem__("rho", d["q2"]["rho"] + 0.02))), True)
    case("M2 Brennan CI lower -0.01", lambda: setattr(R, "_json", with_json(
        lambda rel, d: rel.endswith("concordance_spacefix_brennan_32.json") and d["concordance_ci95"].__setitem__(0, -0.01))), True)
    case("M3 SF.q3_rsa 2->3", lambda: setattr(R, "_json", with_json(
        lambda rel, d: rel.endswith("adr0006_registered_verdicts_spacefix.json") and d.__setitem__("q3_direction_count_rsa", "3/6 instruct > base"))), True)
    case("M4 drop SUPERSEDED A.q2_rho", lambda: R.SUPERSEDED.pop("A.q2_rho"), True)
    case("M5 drop SUPERSEDED cohort.brennan", lambda: R.SUPERSEDED.pop("cohort.brennan"), True)
    case("M6 sibling missing", lambda: R.CLAIMS.__setitem__(slice(None), [c for c in claims0 if c[0] != "SF.q2_rho"]), True)
    R._json = REAL_JSON
    bad = 0
    for name, rc, ok, tail in results:
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:36s} rc={rc}  {' | '.join(t.strip()[:70] for t in tail)}")
    print(f"{len(results) - bad}/{len(results)} mutation cases behaved as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
