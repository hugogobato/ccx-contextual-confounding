"""Micro-example verification (memo section 7) + machinery self-checks.

ME-A: e0 = delta_(0,0), e1 = delta_(1,1): feasible, all witnesses silent.
ME-B: e0 = delta_(0,0), e1 = delta_(0,1): infeasible by exhaustive 16-table
      check; cf1_hard = 1/2; cf1_soft = 0; slack_l1 = 2; degree_signed = 0.
ME-C: feasible non-degenerate point realized by q* = 1/2 a + 1/2 b with
      a = (rX=(0,1), rY=(0,1)), b = (rX=(1,0), rY=(1,1)): slack 0, t* = 1.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import build_iv_A, iv_vertices, discover_facets, facets_feasible, classify_trivial  # noqa: E402
from witnesses import slack_and_feasible, cf1_soft, degree_signed, kl_contextuality, iv_cf1_hard, iv_inflation2_feasible  # noqa: E402


def e_of(pairs):
    """Build conditional observable from dict {(z,x,y): p} with per-z blocks."""
    v = np.zeros(8)
    for (z, x, y), p in pairs.items():
        v[4 * z + 2 * x + y] = p
    return v


def main():
    A = build_iv_A()
    fails = []

    def check(name, cond):
        print(("PASS" if cond else "FAIL"), name)
        if not cond:
            fails.append(name)

    # ---------------- ME-A
    eA = e_of({(0, 0, 0): 1.0, (1, 1, 1): 1.0})
    sA, fA = slack_and_feasible(A, eA)
    check("ME-A feasible", fA)
    check("ME-A slack==0", abs(sA) < 1e-9)
    tA = cf1_soft(A, eA)
    check("ME-A cf1_soft==1", abs(tA - 1.0) < 1e-9)
    dA = degree_signed(A, eA)
    check("ME-A degree_signed==0", abs(dA) < 1e-9)

    # ---------------- ME-B exhaustive infeasibility over 16 tables
    eB = e_of({(0, 0, 0): 1.0, (1, 0, 1): 1.0})
    hit = False
    tabs_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for ix in range(4):
        for iy in range(4):
            ok = True
            for z in (0, 1):
                x = tabs_x[ix][z]
                y = [(0, 0), (0, 1), (1, 0), (1, 1)][iy][x]
                if eB[4 * z + 2 * x + y] == 0:
                    ok = False
            hit |= ok
    check("ME-B no deterministic table on supports", not hit)
    sB, fB = slack_and_feasible(A, eB)
    check("ME-B infeasible (LP)", not fB)
    check("ME-B slack==2 (hand value)", abs(sB - 2.0) < 1e-7)
    tB = cf1_soft(A, eB)
    check("ME-B cf1_soft==0", tB < 1e-9)
    dB = degree_signed(A, eB)
    check("ME-B degree_signed==0 (quasi-coupling)", dB < 1e-9)
    hB = iv_cf1_hard(eB, A)
    check("ME-B cf1_hard==0.5", abs(hB - 0.5) < 1e-12)
    kB = kl_contextuality(A, eB)
    check("ME-B KL == 2*log2 (hand value)", abs(kB - 2 * np.log(2)) < 1e-8)

    # ---------------- ME-C feasible non-degenerate point
    a_col = A[:, 4 * 1 + 1]   # rX=(0,1) idx? careful below
    # explicit tables: a: rX=(0,1) -> ix=1 ((0,1)); rY=(0,1) -> iy=1
    qa = np.zeros(16); qa[4 * 1 + 1] = 0.5
    # b: rX=(1,0) -> ix=2; rY=(1,1) -> iy=3
    qb = np.zeros(16); qb[4 * 2 + 3] = 0.5
    qC = qa + qb
    eC = A.astype(float) @ qC
    sC, fC = slack_and_feasible(A, eC)
    check("ME-C feasible", fC and sC < 1e-9)
    tC = cf1_soft(A, eC)
    check("ME-C cf1_soft==1", abs(tC - 1.0) < 1e-9)
    # memo ME-C: e0 = 1/2 d_(0,0) + 1/2 d_(1,1); e1 = 1/2 d_(1,1) + 1/2 d_(0,1)
    expC = np.zeros(8)
    expC[[0, 3, 5, 7]] = 0.5
    check("ME-C matches memo pushforward", np.allclose(eC, expC))

    # ---------------- facet machinery self-consistency
    H, b = discover_facets(A, seed=0)
    V = iv_vertices()
    check("facets valid at all 16 vertices",
          np.all(H.astype(float) @ V.T <= b[:, None].astype(float) + 1e-9))
    trivial = classify_trivial(H, b, A)
    print(f"facets total={len(H)}, trivial(positivity)={int(trivial.sum())}, "
          f"nontrivial={int((~trivial).sum())}")
    print("nontrivial rows:")
    for h, bb, tr in zip(H, b, trivial):
        if not tr:
            print("  ", "".join(str(x) for x in h), "<=", bb)
    for e, nm in [(eA, "ME-A"), (eB, "ME-B"), (eC, "ME-C")]:
        ff = facets_feasible(H, b, e)
        _, fl = slack_and_feasible(A, e)
        check(f"{nm}: facet-system decision == LP decision", ff == fl)

    # random agreement audit (dictionary pre-check before the big scan)
    rng = np.random.default_rng(7)
    n_agree, n_border = 0, 0
    N = 500
    for _ in range(N):
        e = np.concatenate([rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))])
        ff = facets_feasible(H, b, e)
        _, fl = slack_and_feasible(A, e)
        s, _ = slack_and_feasible(A, e)
        if min(s, 1.0) <= s and 1e-9 < s < 1e-7:
            n_border += 1
            continue
        n_agree += int(ff == fl)
    print(f"audit agreement {n_agree}/{N - n_border} (borderline flagged: {n_border})")
    check("audit full agreement", n_agree == N - n_border)

    # inflation-2 cross-check on micro examples
    check("ME-A inflation2 feasible", iv_inflation2_feasible(eA, A))
    check("ME-B inflation2 infeasible", not iv_inflation2_feasible(eB, A))
    check("ME-C inflation2 feasible", iv_inflation2_feasible(eC, A))

    print()
    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("ALL MICRO TESTS PASSED")


if __name__ == "__main__":
    main()
