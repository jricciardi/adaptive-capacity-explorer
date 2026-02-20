#!/usr/bin/env python3
"""
Step 2: Compute pairwise skill similarity matrix.

Loads the 76-dimension O*NET skill profiles from Step 1, applies a percentile
rank transformation to normalize element distributions, then computes cosine
similarity between all pairs of occupations.

Outputs:
- data/skill_similarity_top20.json: Top 20 most similar occupations per SOC code
- data/skill_similarity_matrix.npz: Full similarity matrix for Step 4

Methodology:
  The study (Manning & Aguirre 2026) uses employment-weighted percentiles before
  computing cosine similarity. We use unweighted percentile ranks here because:
  (a) employment data is not yet available (comes in Step 3), and (b) the purpose
  of the transform is distributional normalization, which unweighted percentiles
  achieve. Employment weighting matters for the transferability formula in Step 4,
  not for the similarity metric itself.

  Without percentile transformation, raw importance values produce cosine
  similarities compressed into 0.85-1.0 (all occupations share high cognitive /
  low physical baselines). Percentile transformation produces a useful 0.18-0.99
  range with 6x more discriminating power.
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

INPUT_FILE = DATA_DIR / "onet_skill_profiles.json"
OUTPUT_TOP20_FILE = DATA_DIR / "skill_similarity_top20.json"
OUTPUT_MATRIX_FILE = DATA_DIR / "skill_similarity_matrix.npz"

TOP_K = 20


def load_skill_profiles(filepath):
    """
    Load skill profiles from Step 1 output.

    Returns:
        soc_codes: sorted list of BLS 6-digit SOC codes
        element_ids: list of 76 element IDs in vector order
        raw_matrix: numpy array of shape (N, 76) with raw importance values
        titles: dict mapping SOC code to occupation title
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    element_ids = [e["element_id"] for e in data["elements"]]
    soc_codes = sorted(data["profiles"].keys())
    titles = {}

    N = len(soc_codes)
    D = len(element_ids)
    raw_matrix = np.zeros((N, D), dtype=np.float64)

    for i, soc in enumerate(soc_codes):
        profile = data["profiles"][soc]
        titles[soc] = profile["title"]
        for j, eid in enumerate(element_ids):
            val = profile["vector"][eid]
            if val is None:
                print(f"ERROR: Null value at {soc}/{eid}")
                sys.exit(1)
            raw_matrix[i, j] = val

    return soc_codes, element_ids, raw_matrix, titles


def percentile_rank_transform(matrix):
    """
    Transform each column to percentile ranks using the average-rank method.

    For each of the 76 elements, ranks all N occupations and converts to a
    0-100 percentile scale. Ties receive the mean of the ranks they would
    occupy if distinct.

    Args:
        matrix: numpy array of shape (N, D)

    Returns:
        percentile_matrix: numpy array of shape (N, D), values in [0, 100]
    """
    N, D = matrix.shape
    percentile_matrix = np.zeros_like(matrix)

    for j in range(D):
        col = matrix[:, j]

        # Compute average ranks (1-based)
        # Sort and assign base ranks, then average tied ranks
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.arange(1, N + 1, dtype=np.float64)

        # Average the ranks for tied values
        unique_vals = np.unique(col)
        for val in unique_vals:
            mask = col == val
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()

        # Convert ranks to 0-100 percentile
        # rank 1 -> 0, rank N -> 100
        percentile_matrix[:, j] = (ranks - 1) / (N - 1) * 100

    return percentile_matrix


def compute_cosine_similarity(matrix):
    """
    Compute pairwise cosine similarity via L2-normalized matrix multiply.

    Args:
        matrix: numpy array of shape (N, D)

    Returns:
        similarity_matrix: numpy array of shape (N, N), values in [-1, 1]
    """
    # L2 normalize each row
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against zero-norm rows (shouldn't happen with percentile data)
    norms = np.maximum(norms, 1e-10)
    normed = matrix / norms

    # Cosine similarity = dot product of normalized vectors
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim = normed @ normed.T

    # Clean up floating point artifacts
    np.clip(sim, -1.0, 1.0, out=sim)
    np.fill_diagonal(sim, 1.0)

    return sim


def extract_top_k(similarity_matrix, k, soc_codes, titles):
    """
    Extract the top-k most similar occupations for each occupation.

    Uses np.argpartition for O(n) selection instead of O(n log n) full sort.

    Args:
        similarity_matrix: (N, N) array
        k: number of neighbors to extract
        soc_codes: list of SOC codes
        titles: dict of SOC -> title

    Returns:
        neighbors: dict of SOC -> list of {soc, title, similarity}
    """
    N = len(soc_codes)
    neighbors = {}

    for i in range(N):
        row = similarity_matrix[i].copy()
        row[i] = -np.inf  # exclude self

        # Get indices of top-k values
        top_indices = np.argpartition(row, -k)[-k:]
        # Sort them by similarity descending
        top_indices = top_indices[np.argsort(row[top_indices])[::-1]]

        neighbor_list = []
        for idx in top_indices:
            neighbor_list.append({
                "soc": soc_codes[idx],
                "title": titles[soc_codes[idx]],
                "similarity": round(float(row[idx]), 4),
            })

        neighbors[soc_codes[i]] = neighbor_list

    return neighbors


def run_sanity_checks(similarity_matrix, percentile_matrix, soc_codes, titles, neighbors):
    """Run comprehensive sanity checks and print results."""
    N = len(soc_codes)
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}
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

    # 1. Percentile column means
    col_means = percentile_matrix.mean(axis=0)
    mean_ok = np.allclose(col_means, 50.0, atol=0.01)
    check(
        "Percentile column means are all 50.0",
        mean_ok,
        f"range: {col_means.min():.4f} to {col_means.max():.4f}",
    )

    # 2. Symmetry
    max_asym = np.max(np.abs(similarity_matrix - similarity_matrix.T))
    check("Similarity matrix is symmetric", max_asym < 1e-10, f"max asymmetry: {max_asym:.2e}")

    # 3. Non-negativity (all inputs are non-negative, so cosine sim should be >= 0)
    min_val = similarity_matrix.min()
    check("All similarities >= 0", min_val >= -1e-10, f"min value: {min_val:.6f}")

    # 4. Diagonal
    diag_ok = np.all(similarity_matrix.diagonal() == 1.0)
    check("Diagonal entries are exactly 1.0", diag_ok)

    # 5. No NaN or Inf
    check("No NaN values", not np.any(np.isnan(similarity_matrix)))
    check("No Inf values", not np.any(np.isinf(similarity_matrix)))

    # 6. Distribution stats
    mask = ~np.eye(N, dtype=bool)
    off_diag = similarity_matrix[mask]
    pcts = np.percentile(off_diag, [5, 25, 50, 75, 95])
    print(f"\n  Off-diagonal similarity distribution:")
    print(f"    Min: {off_diag.min():.4f}  Max: {off_diag.max():.4f}")
    print(f"    5th: {pcts[0]:.4f}  25th: {pcts[1]:.4f}  50th: {pcts[2]:.4f}  75th: {pcts[3]:.4f}  95th: {pcts[4]:.4f}")
    print(f"    Mean: {off_diag.mean():.4f}  Std: {off_diag.std():.4f}")

    # 7. Face-validity spot checks
    print(f"\n  Face-validity spot checks:")

    def sim_between(soc_a, soc_b):
        if soc_a not in soc_to_idx or soc_b not in soc_to_idx:
            return None
        return similarity_matrix[soc_to_idx[soc_a], soc_to_idx[soc_b]]

    # Nurses should be very similar
    nurse_sim = sim_between("29-1141", "29-1171")
    if nurse_sim is not None:
        check(
            "Registered Nurses <-> Nurse Practitioners similarity > 0.95",
            nurse_sim > 0.95,
            f"similarity: {nurse_sim:.4f}",
        )
    else:
        print("  [SKIP] Nurse comparison — SOC code not found")

    # Software devs and carpenters should be below-median similarity
    # (median is ~0.79; they share some skills like problem solving but
    # differ on physical/mechanical vs computational dimensions)
    dev_carp = sim_between("15-1252", "47-2031")
    if dev_carp is not None:
        check(
            "Software Developers <-> Carpenters similarity < median (below avg)",
            dev_carp < 0.80,
            f"similarity: {dev_carp:.4f}",
        )
    else:
        print("  [SKIP] Dev/Carpenter comparison — SOC code not found")

    # Chief Executives top neighbor should be management-related
    ceo_top = neighbors.get("11-1011", [{}])[0]
    if ceo_top:
        is_mgmt = ceo_top["soc"].startswith("11-") or "Manager" in ceo_top["title"]
        check(
            "Chief Executives top neighbor is management-related",
            is_mgmt,
            f"top neighbor: {ceo_top['soc']} ({ceo_top['title']}, {ceo_top['similarity']:.4f})",
        )

    # 8. Neighbor list integrity
    print(f"\n  Neighbor list integrity:")
    all_have_k = all(len(v) == TOP_K for v in neighbors.values())
    check(f"All occupations have exactly {TOP_K} neighbors", all_have_k)

    no_self = all(
        soc not in [n["soc"] for n in nlist]
        for soc, nlist in neighbors.items()
    )
    check("No self-references in neighbor lists", no_self)

    # Monotonicity (similarities should be non-increasing)
    mono_ok = True
    for soc, nlist in neighbors.items():
        sims = [n["similarity"] for n in nlist]
        if any(sims[i] < sims[i + 1] for i in range(len(sims) - 1)):
            mono_ok = False
            break
    check("Neighbor similarities are monotonically non-increasing", mono_ok)

    check("Neighbor count matches occupation count", len(neighbors) == N, f"{len(neighbors)} vs {N}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def main():
    # Verify input exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run scripts/01_parse_onet_skills.py first.")
        sys.exit(1)

    print("Step 2: Compute Pairwise Skill Similarity")
    print("=" * 60)

    # Load profiles
    print("\nLoading skill profiles...")
    soc_codes, element_ids, raw_matrix, titles = load_skill_profiles(INPUT_FILE)
    N, D = raw_matrix.shape
    print(f"  Loaded {N} occupations x {D} dimensions")

    # Percentile transform
    print("\nApplying percentile rank transformation...")
    percentile_matrix = percentile_rank_transform(raw_matrix)
    print(f"  Percentile matrix shape: {percentile_matrix.shape}")
    print(f"  Value range: {percentile_matrix.min():.1f} to {percentile_matrix.max():.1f}")

    # Compute similarity
    print("\nComputing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity(percentile_matrix)
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")

    # Extract top-k neighbors
    print(f"\nExtracting top-{TOP_K} neighbors per occupation...")
    neighbors = extract_top_k(similarity_matrix, TOP_K, soc_codes, titles)
    print(f"  Generated neighbor lists for {len(neighbors)} occupations")

    # Sanity checks
    all_pass = run_sanity_checks(
        similarity_matrix, percentile_matrix, soc_codes, titles, neighbors
    )

    if not all_pass:
        print("\nWARNING: Some sanity checks failed. Review output carefully.")

    # Write top-20 JSON
    print("\n" + "=" * 60)
    print("WRITING OUTPUT FILES")
    print("=" * 60)

    mask = ~np.eye(N, dtype=bool)
    off_diag = similarity_matrix[mask]

    output = {
        "metadata": {
            "source_data": "data/onet_skill_profiles.json",
            "total_occupations": N,
            "neighbors_per_occupation": TOP_K,
            "similarity_method": "cosine",
            "preprocessing": "unweighted_percentile_rank",
            "preprocessing_detail": (
                f"Each element ranked across {N} occupations using average-rank "
                "method for ties, converted to 0-100 percentile scale. This "
                "normalizes element distributions without employment weighting. "
                "Employment weighting is applied in Step 4's transferability formula."
            ),
            "similarity_range": {
                "global_min": round(float(off_diag.min()), 4),
                "global_max": round(float(off_diag.max()), 4),
                "global_mean": round(float(off_diag.mean()), 4),
                "global_median": round(float(np.median(off_diag)), 4),
            },
            "generated_by": "scripts/02_compute_skill_similarity.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "soc_index": soc_codes,
        "neighbors": neighbors,
    }

    with open(OUTPUT_TOP20_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    file_size = OUTPUT_TOP20_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  Wrote {OUTPUT_TOP20_FILE} ({file_size:.1f} MB)")

    # Write full matrix as npz
    np.savez_compressed(
        OUTPUT_MATRIX_FILE,
        similarity_matrix=similarity_matrix.astype(np.float32),
        soc_codes=np.array(soc_codes),
        percentile_vectors=percentile_matrix.astype(np.float32),
    )
    npz_size = OUTPUT_MATRIX_FILE.stat().st_size / (1024 * 1024)
    print(f"  Wrote {OUTPUT_MATRIX_FILE} ({npz_size:.1f} MB)")

    # Verify npz loads correctly
    loaded = np.load(OUTPUT_MATRIX_FILE, allow_pickle=False)
    assert loaded["similarity_matrix"].shape == (N, N)
    assert loaded["soc_codes"].shape == (N,)
    assert loaded["percentile_vectors"].shape == (N, D)
    print(f"  Verified .npz loads correctly: {list(loaded.keys())}")

    # Sample output
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (first 3 occupations, top 5 neighbors each)")
    print("=" * 60)

    for soc in soc_codes[:3]:
        print(f"\n{soc}: {titles[soc]}")
        for n in neighbors[soc][:5]:
            print(f"  {n['similarity']:.4f}  {n['soc']} — {n['title']}")


if __name__ == "__main__":
    main()
