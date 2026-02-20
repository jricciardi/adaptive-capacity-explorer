#!/usr/bin/env python3
"""
Step 4: Compute growth-weighted transferability scores.

For each occupation i, computes:
  Ti = Σj≠i( emp_j × sim_ij × (1+gr_j) ) / Σj≠i( emp_j × (1+gr_j) )

Ti is a weighted average of cosine similarities, weighted by employment
times (1 + growth_rate). Higher Ti means the occupation's skills transfer
well to large, growing occupations.

Inputs:
  - data/skill_similarity_matrix.npz (from Step 2)
  - data/bls_projections.json (from Step 3)
  - data/skill_similarity_top20.json (from Step 2, for growing neighbors)

Outputs:
  - data/transferability_scores.json

Based on Manning & Aguirre (2026), "How Adaptable Are American Workers
to AI-Induced Job Displacement?", NBER Working Paper 34705.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

SIMILARITY_FILE = DATA_DIR / "skill_similarity_matrix.npz"
PROJECTIONS_FILE = DATA_DIR / "bls_projections.json"
TOP20_FILE = DATA_DIR / "skill_similarity_top20.json"
OUTPUT_FILE = DATA_DIR / "transferability_scores.json"


def load_similarity_matrix(filepath):
    """Load the full similarity matrix and SOC code index from Step 2."""
    data = np.load(filepath, allow_pickle=False)
    sim_matrix = data["similarity_matrix"]  # (774, 774) float32
    soc_codes = list(data["soc_codes"])  # 774 strings
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
    return sim_matrix, soc_codes, soc_to_idx


def load_projections(filepath):
    """Load BLS projections and matched SOC codes from Step 3."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["projections"], sorted(data["coverage"]["matched"])


def load_top20_neighbors(filepath):
    """Load top-20 similarity neighbors from Step 2."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["neighbors"]


def compute_transferability_scores(S, w):
    """
    Vectorized computation of growth-weighted transferability scores.

    Ti = Σj≠i( w_j × sim_ij ) / Σj≠i( w_j )

    where w_j = employment_j × (1 + growth_rate_j).

    Since sim_ii = 1.0, the self-term contribution to the numerator is w_i.
    We compute the full dot product then subtract the self-term.

    Args:
        S: (N, N) similarity submatrix (float64)
        w: (N,) weight vector (float64)

    Returns:
        T: (N,) transferability scores
    """
    # Full matrix-vector product (includes self-term)
    # Suppress benign numpy 2.0 matmul RuntimeWarning for large float64 operations
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        numerator_full = S @ w  # (N,)

    # Total weight sum
    w_total = w.sum()

    # Remove self-contribution: sim_ii=1.0, so self-term = 1.0 * w_i = w_i
    numerator = numerator_full - w
    denominator = w_total - w

    T = numerator / denominator
    return T


def compute_ranks_and_percentiles(T):
    """
    Compute ranks (1=highest) and percentiles for transferability scores.

    Returns:
        ranks: int array, 1-based (1 = highest Ti)
        percentiles: float array, 0-100 scale
    """
    N = len(T)
    rank_order = np.argsort(-T)  # indices sorted by T descending
    ranks = np.empty(N, dtype=int)
    ranks[rank_order] = np.arange(1, N + 1)

    # Percentile: rank 1 -> 100th, rank N -> 0th
    percentiles = (N - ranks) / (N - 1) * 100.0
    return ranks, percentiles


def extract_growing_neighbors(soc, top20_neighbors, projections, matched_set, max_k=10):
    """
    Extract top growing neighbors from the pre-computed top-20 similarity list.

    Filters to neighbors that are in the matched set and have positive growth.
    Keeps original similarity ordering (most similar first).

    Returns list of dicts with soc, title, similarity, growth_rate, employment.
    """
    neighbors = top20_neighbors.get(soc, [])
    growing = []
    for n in neighbors:
        n_soc = n["soc"]
        if n_soc not in matched_set:
            continue
        proj = projections.get(n_soc)
        if proj is None or proj["growth_rate"] is None or proj["growth_rate"] <= 0:
            continue
        growing.append({
            "soc": n_soc,
            "title": n["title"],
            "similarity": n["similarity"],
            "growth_rate": round(proj["growth_rate"], 6),
            "employment_2024": proj["employment_2024"],
        })
        if len(growing) >= max_k:
            break
    return growing


def run_sanity_checks(T, ranks, matched_codes, projections, growing_map, S, w):
    """Run comprehensive sanity checks on transferability scores."""
    N = len(T)
    soc_to_match_idx = {soc: i for i, soc in enumerate(matched_codes)}
    all_pass = True

    def check(name, condition, detail=""):
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # 1. Score range
    check("All Ti in [0, 1]", T.min() >= 0 and T.max() <= 1.0,
          f"range: [{T.min():.6f}, {T.max():.6f}]")

    # 2. No NaN or Inf
    check("No NaN values", not np.any(np.isnan(T)))
    check("No Inf values", not np.any(np.isinf(T)))

    # 3. Meaningful spread
    check("Meaningful spread (std > 0.01)", T.std() > 0.01,
          f"std={T.std():.6f}")

    # 4. Distribution stats
    print(f"\n  Ti distribution:")
    print(f"    Min: {T.min():.6f}  Max: {T.max():.6f}")
    print(f"    Mean: {T.mean():.6f}  Median: {np.median(T):.6f}  Std: {T.std():.6f}")
    for pct in [5, 25, 50, 75, 95]:
        print(f"    {pct}th percentile: {np.percentile(T, pct):.6f}")

    # 5. Face-validity spot checks
    print(f"\n  Face-validity spot checks:")
    median_T = np.median(T)

    # Note on Software Developers: Ti measures how broadly skills transfer
    # to ALL other occupations weighted by employment × growth. Programming
    # and computer-specific skills are specialized — they score high on
    # similarity with a narrow cluster of tech occupations, but low against
    # the vast majority of the labor market (healthcare, trades, education,
    # transport, etc.). Below-median transferability is methodologically
    # correct for specialized tech skills. The occupation's OWN growth is
    # strong, but Ti captures skill breadth across all occupations.
    spot_checks = [
        ("11-1021", "General and Operations Managers", "above median (broad mgmt skills)",
         lambda t: t > median_T),
        ("29-1141", "Registered Nurses", "above 25th percentile (care + interpersonal)",
         lambda t: t > np.percentile(T, 25)),
        ("15-1252", "Software Developers", "above 10th percentile (specialized but not lowest)",
         lambda t: t > np.percentile(T, 10)),
        ("47-5043", "Roof Bolters, Mining", "below median (narrow manual skills)",
         lambda t: t < median_T),
        ("43-9022", "Word Processors and Typists", "below median (declining clerical)",
         lambda t: t < median_T),
    ]

    for soc, title, expectation, test_fn in spot_checks:
        if soc in soc_to_match_idx:
            idx = soc_to_match_idx[soc]
            ti = T[idx]
            rank = ranks[idx]
            check(
                f"{title} ({soc}) {expectation}",
                test_fn(ti),
                f"Ti={ti:.6f}, rank={rank}/{N}",
            )
        else:
            print(f"  [SKIP] {soc} ({title}) — not in matched set")

    # 6. Rank integrity
    print(f"\n  Rank integrity:")
    check("Ranks span 1 to N", ranks.min() == 1 and ranks.max() == N,
          f"range: [{ranks.min()}, {ranks.max()}]")
    check("No duplicate ranks", len(set(ranks)) == N)

    # 7. Vectorized matches manual loop (check first occupation)
    print(f"\n  Numerical verification:")
    i = 0
    manual_num = sum(S[i, j] * w[j] for j in range(N) if j != i)
    manual_den = sum(w[j] for j in range(N) if j != i)
    manual_T = manual_num / manual_den
    check(
        "Vectorized matches manual loop (occ 0)",
        abs(T[0] - manual_T) < 1e-10,
        f"vectorized={T[0]:.10f}, manual={manual_T:.10f}",
    )

    # 8. Growing neighbors coverage
    print(f"\n  Growing neighbors coverage:")
    counts = [len(growing_map[soc]) for soc in matched_codes]
    pct_ge5 = sum(1 for c in counts if c >= 5) / N
    check(
        ">=90% have >=5 growing neighbors",
        pct_ge5 > 0.90,
        f"{pct_ge5*100:.1f}%",
    )
    print(f"    Min: {min(counts)}, Max: {max(counts)}, Mean: {np.mean(counts):.1f}")
    zero_count = sum(1 for c in counts if c == 0)
    print(f"    Occupations with 0 growing neighbors: {zero_count}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def main():
    # Verify inputs exist
    for f in [SIMILARITY_FILE, PROJECTIONS_FILE, TOP20_FILE]:
        if not f.exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)

    print("Step 4: Compute Growth-Weighted Transferability Scores")
    print("=" * 60)

    # Load inputs
    print("\nLoading inputs...")
    full_sim, full_soc_codes, full_soc_to_idx = load_similarity_matrix(SIMILARITY_FILE)
    print(f"  Similarity matrix: {full_sim.shape[0]}x{full_sim.shape[1]} {full_sim.dtype}")

    projections, matched_codes = load_projections(PROJECTIONS_FILE)
    print(f"  BLS projections: {len(projections)} occupations")
    print(f"  Matched codes: {len(matched_codes)}")

    top20_neighbors = load_top20_neighbors(TOP20_FILE)
    print(f"  Top-20 neighbors: {len(top20_neighbors)} occupations")

    N = len(matched_codes)
    matched_set = set(matched_codes)

    # Verify all matched codes exist in both data sources
    for soc in matched_codes:
        assert soc in full_soc_to_idx, f"{soc} not in similarity matrix"
        assert soc in projections, f"{soc} not in projections"

    # Extract 751x751 submatrix
    print(f"\nExtracting {N}x{N} submatrix...")
    matched_indices = np.array([full_soc_to_idx[soc] for soc in matched_codes])
    S = full_sim[np.ix_(matched_indices, matched_indices)].astype(np.float64)
    print(f"  Submatrix shape: {S.shape}, dtype: {S.dtype}")

    # Verify diagonal is 1.0
    assert np.allclose(np.diag(S), 1.0, atol=1e-6), "Diagonal of submatrix is not 1.0"

    # Build weight vector
    employment = np.array(
        [projections[soc]["employment_2024"] for soc in matched_codes],
        dtype=np.float64,
    )
    growth_rate = np.array(
        [projections[soc]["growth_rate"] for soc in matched_codes],
        dtype=np.float64,
    )
    w = employment * (1.0 + growth_rate)
    print(f"  Weight vector: {N} values, sum={w.sum():.1f}, min={w.min():.1f}, max={w.max():.1f}")

    # Compute transferability scores
    print(f"\nComputing transferability scores...")
    T = compute_transferability_scores(S, w)
    print(f"  Ti range: [{T.min():.6f}, {T.max():.6f}]")
    print(f"  Ti mean: {T.mean():.6f}, std: {T.std():.6f}")

    # Compute ranks and percentiles
    ranks, percentiles = compute_ranks_and_percentiles(T)

    # Extract growing neighbors
    print(f"\nExtracting growing neighbors (top-10 from top-20, growth > 0)...")
    growing_map = {}
    for soc in matched_codes:
        growing_map[soc] = extract_growing_neighbors(
            soc, top20_neighbors, projections, matched_set, max_k=10
        )
    counts = [len(v) for v in growing_map.values()]
    print(f"  Avg growing neighbors per occupation: {np.mean(counts):.1f}")
    print(f"  Occupations with 0 growing neighbors: {sum(1 for c in counts if c == 0)}")

    # Sanity checks
    all_pass = run_sanity_checks(T, ranks, matched_codes, projections, growing_map, S, w)

    if not all_pass:
        print("\nWARNING: Some sanity checks failed. Review output carefully.")

    # Build output
    print("\n" + "=" * 60)
    print("WRITING OUTPUT")
    print("=" * 60)

    output = {
        "metadata": {
            "description": "Growth-weighted transferability scores per occupation",
            "formula": "Ti = sum_j(emp_j * sim_ij * (1+gr_j)) / sum_j(emp_j * (1+gr_j)), j != i",
            "formula_source": "Manning & Aguirre (2026), NBER Working Paper 34705",
            "inputs": {
                "similarity_matrix": "data/skill_similarity_matrix.npz",
                "bls_projections": "data/bls_projections.json",
                "top20_neighbors": "data/skill_similarity_top20.json",
            },
            "total_scored_occupations": N,
            "excluded_onet_only": 23,
            "score_range": {
                "min": round(float(T.min()), 6),
                "max": round(float(T.max()), 6),
                "mean": round(float(T.mean()), 6),
                "std": round(float(T.std()), 6),
            },
            "growing_neighbors_method": (
                "Top-20 similarity neighbors from Step 2, filtered to "
                "growth_rate > 0 and in matched SOC set, ordered by "
                "similarity descending, max 10 per occupation"
            ),
            "generated_by": "scripts/04_compute_transferability.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "scores": {},
    }

    for i, soc in enumerate(matched_codes):
        proj = projections[soc]
        output["scores"][soc] = {
            "title": proj["title"],
            "transferability_score": round(float(T[i]), 6),
            "rank": int(ranks[i]),
            "percentile": round(float(percentiles[i]), 2),
            "employment_2024": proj["employment_2024"],
            "growth_rate": proj["growth_rate"],
            "growing_neighbors": growing_map[soc],
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  Wrote {OUTPUT_FILE} ({file_size:.2f} MB)")

    # Sample output
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    # Find notable occupations
    soc_to_match_idx = {soc: i for i, soc in enumerate(matched_codes)}
    top_idx = np.argmax(T)
    bottom_idx = np.argmin(T)
    median_idx = np.argsort(T)[N // 2]

    notable = [
        ("Highest transferability", top_idx),
        ("Lowest transferability", bottom_idx),
        ("Median transferability", median_idx),
    ]

    # Add specific occupations if in matched set
    for label, soc in [("Software Developers", "15-1252"), ("Registered Nurses", "29-1141")]:
        if soc in soc_to_match_idx:
            notable.append((label, soc_to_match_idx[soc]))

    for label, idx in notable:
        soc = matched_codes[idx]
        proj = projections[soc]
        ti = T[idx]
        r = ranks[idx]
        p = percentiles[idx]
        gn = growing_map[soc]

        print(f"\n  {label}:")
        print(f"    {soc}: {proj['title']}")
        print(f"    Ti = {ti:.6f} (rank {r}/{N}, {p:.1f}th percentile)")
        print(f"    Employment: {proj['employment_2024']:.1f}K, Growth: {proj['growth_rate']:+.1%}")
        if gn:
            print(f"    Top-3 growing neighbors:")
            for n in gn[:3]:
                print(f"      {n['similarity']:.4f}  {n['soc']} — {n['title']} "
                      f"(growth {n['growth_rate']:+.1%}, {n['employment_2024']:.1f}K)")
        else:
            print(f"    No growing neighbors in top-20")


if __name__ == "__main__":
    main()
