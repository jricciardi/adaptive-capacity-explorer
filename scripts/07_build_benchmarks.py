#!/usr/bin/env python3
"""
Step 7: Build occupation-level benchmark table.

Combines data from Steps 3–6 (BLS projections, transferability, density,
AI exposure) into a single front-end-ready JSON file. Also computes a
2-component partial composite score using Manning & Aguirre's methodology:
winsorize → employment-weighted Z-scores → average → percentile rank.

The composite uses transferability and density (the 2 components available at
the occupation level). Wealth and age are user-inputted at runtime in the
front-end, which will extend this to the full 4-component score.

Inputs:
  - data/bls_projections.json (from Step 3)
  - data/transferability_scores.json (from Step 4)
  - data/density_scores.json (from Step 5)
  - data/ai_exposure.json (from Step 6)

Outputs:
  - data/occupation_benchmarks.json

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

BLS_FILE = DATA_DIR / "bls_projections.json"
TRANSFER_FILE = DATA_DIR / "transferability_scores.json"
DENSITY_FILE = DATA_DIR / "density_scores.json"
EXPOSURE_FILE = DATA_DIR / "ai_exposure.json"
OUTPUT_FILE = DATA_DIR / "occupation_benchmarks.json"


def load_json(filepath, label):
    """Load a JSON file with error handling."""
    if not filepath.exists():
        print(f"ERROR: {label} not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def employment_weighted_percentile(values, weights, p):
    """
    Compute the p-th employment-weighted percentile.

    Uses linear interpolation between the two values whose cumulative
    weight brackets the target percentile.

    Args:
        values: array of component values
        weights: array of employment weights (same length)
        p: percentile in [0, 100]

    Returns:
        The weighted percentile value
    """
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_weights = weights[order]

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]

    # Target cumulative weight for percentile p
    target = p / 100.0 * total_weight

    # Find the value where cumulative weight crosses the target
    idx = np.searchsorted(cum_weights, target)
    idx = min(idx, len(sorted_vals) - 1)

    return sorted_vals[idx]


def winsorize_weighted(values, weights, lower_pct=1, upper_pct=99):
    """
    Winsorize values at employment-weighted percentiles.

    Args:
        values: array of component values
        weights: array of employment weights
        lower_pct: lower percentile cutoff (default 1)
        upper_pct: upper percentile cutoff (default 99)

    Returns:
        winsorized: clipped array
        p_low: lower cutoff value
        p_high: upper cutoff value
    """
    p_low = employment_weighted_percentile(values, weights, lower_pct)
    p_high = employment_weighted_percentile(values, weights, upper_pct)

    winsorized = np.clip(values, p_low, p_high)
    return winsorized, p_low, p_high


def weighted_zscore(values, weights):
    """
    Compute employment-weighted Z-scores.

    Z = (x - weighted_mean) / weighted_std

    Args:
        values: array of values
        weights: array of employment weights

    Returns:
        z_scores: array of Z-scores
        w_mean: weighted mean
        w_std: weighted standard deviation
    """
    total_w = weights.sum()
    w_mean = np.sum(weights * values) / total_w
    w_var = np.sum(weights * (values - w_mean) ** 2) / total_w
    w_std = np.sqrt(w_var)

    if w_std < 1e-10:
        print("  WARNING: Near-zero weighted std; Z-scores will all be ~0")
        return np.zeros_like(values), w_mean, w_std

    z_scores = (values - w_mean) / w_std
    return z_scores, w_mean, w_std


def employment_weighted_percentile_rank(values, weights):
    """
    Compute employment-weighted percentile ranks (0–100 scale).

    For each occupation i, the percentile rank is the fraction of total
    employment in occupations with a lower composite score, expressed as
    a percentage.

    Args:
        values: array of composite Z-scores
        weights: array of employment weights

    Returns:
        percentile_ranks: array of percentile ranks (0–100)
    """
    N = len(values)
    total_w = weights.sum()
    percentile_ranks = np.zeros(N)

    for i in range(N):
        # Employment share with strictly lower composite
        below = np.sum(weights[values < values[i]])
        # Employment share with equal composite (for tie handling)
        equal = np.sum(weights[values == values[i]])
        # Percentile = (below + 0.5 * equal) / total * 100
        # The 0.5 * equal gives mid-rank for ties
        percentile_ranks[i] = (below + 0.5 * equal) / total_w * 100

    return percentile_ranks


def run_sanity_checks(benchmarks, composite_socs, all_socs):
    """Run comprehensive sanity checks and print results."""
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

    # 1. Total occupations
    check("Total occupations = 751", len(all_socs) == 751, f"{len(all_socs)}")

    # 2. Full data count
    full_data = [soc for soc in all_socs if benchmarks[soc].get("density_available", False)]
    check("Occupations with full data = 740", len(full_data) == 740, f"{len(full_data)}")

    # 3. Composite count
    with_composite = [soc for soc in all_socs if benchmarks[soc].get("composite_score") is not None]
    check("Occupations with composite = 740", len(with_composite) == 740, f"{len(with_composite)}")

    # 4. No NaN in composite
    composite_vals = [benchmarks[soc]["composite_score"] for soc in with_composite]
    check("No NaN in composite scores", not any(np.isnan(v) for v in composite_vals))

    # 5. Composite percentile range
    min_comp = min(composite_vals)
    max_comp = max(composite_vals)
    check(
        "Composite percentile range spans 0–100",
        min_comp < 5 and max_comp > 95,
        f"range: {min_comp:.2f} to {max_comp:.2f}",
    )

    # 6. Employment-weighted Z-score means ≈ 0
    trans_zs = [benchmarks[soc]["transferability_z"] for soc in composite_socs]
    dens_zs = [benchmarks[soc]["density_z"] for soc in composite_socs]
    emps = [benchmarks[soc]["employment_2024"] for soc in composite_socs]
    total_emp = sum(emps)

    wmean_trans_z = sum(e * z for e, z in zip(emps, trans_zs)) / total_emp
    wmean_dens_z = sum(e * z for e, z in zip(emps, dens_zs)) / total_emp
    check(
        "Weighted mean of transferability Z ≈ 0",
        abs(wmean_trans_z) < 0.01,
        f"{wmean_trans_z:.6f}",
    )
    check(
        "Weighted mean of density Z ≈ 0",
        abs(wmean_dens_z) < 0.01,
        f"{wmean_dens_z:.6f}",
    )

    # 7. All have title
    all_titled = all(benchmarks[soc].get("title") for soc in all_socs)
    check("All occupations have title", all_titled)

    # 8. All have employment
    all_emp = all(benchmarks[soc].get("employment_2024") is not None for soc in all_socs)
    check("All occupations have employment_2024", all_emp)

    # 9. Wage coverage
    wage_count = sum(
        1 for soc in all_socs
        if benchmarks[soc].get("median_annual_wage") is not None
    )
    check(
        "Wage coverage ≥ 700",
        wage_count >= 700,
        f"{wage_count}/{len(all_socs)}",
    )

    # 10. Face validity: General and Operations Managers (11-1021)
    if "11-1021" in benchmarks and benchmarks["11-1021"].get("composite_score") is not None:
        mgr_comp = benchmarks["11-1021"]["composite_score"]
        check(
            "General Managers composite > 60th percentile",
            mgr_comp > 60,
            f"composite: {mgr_comp:.2f}",
        )
    else:
        print("  [SKIP] General Managers — not in composite set")

    # 11. Face validity: Mining/extraction below 40th
    mining_socs = [soc for soc in composite_socs if soc.startswith("47-5")]
    if mining_socs:
        mining_comps = [benchmarks[soc]["composite_score"] for soc in mining_socs]
        avg_mining = np.mean(mining_comps)
        check(
            "Mining/extraction avg composite < 40th percentile",
            avg_mining < 40,
            f"avg: {avg_mining:.2f} (N={len(mining_socs)})",
        )
    else:
        print("  [SKIP] Mining/extraction — none found")

    # 12. Composite correlates with density
    comp_arr = np.array([benchmarks[soc]["composite_score"] for soc in composite_socs])
    dens_arr = np.array([benchmarks[soc]["density_score"] for soc in composite_socs])
    r_dens = np.corrcoef(comp_arr, dens_arr)[0, 1]
    check(
        "Composite correlates positively with density (r > 0.3)",
        r_dens > 0.3,
        f"r = {r_dens:.4f}",
    )

    # 13. Composite correlates with transferability
    trans_arr = np.array([benchmarks[soc]["transferability_score"] for soc in composite_socs])
    r_trans = np.corrcoef(comp_arr, trans_arr)[0, 1]
    check(
        "Composite correlates positively with transferability (r > 0.3)",
        r_trans > 0.3,
        f"r = {r_trans:.4f}",
    )

    # 14. Growing neighbors coverage
    gn_count = sum(
        1 for soc in all_socs
        if benchmarks[soc].get("growing_neighbors") and len(benchmarks[soc]["growing_neighbors"]) > 0
    )
    check(
        "Growing neighbors present for ≥ 700 occupations",
        gn_count >= 700,
        f"{gn_count}/{len(all_socs)}",
    )

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 7: Build Occupation-Level Benchmark Table")
    print("=" * 60)

    # ================================================================
    # 1. Load all input files
    # ================================================================
    print("\n1. Loading input files...")

    bls_data = load_json(BLS_FILE, "BLS projections")
    transfer_data = load_json(TRANSFER_FILE, "Transferability scores")
    density_data = load_json(DENSITY_FILE, "Density scores")
    exposure_data = load_json(EXPOSURE_FILE, "AI exposure scores")

    # Canonical SOC universe: the 751 matched codes
    matched_socs = sorted(bls_data["coverage"]["matched"])
    print(f"  BLS projections: {len(bls_data['projections'])} total, {len(matched_socs)} matched")
    print(f"  Transferability: {len(transfer_data['scores'])} occupations")
    print(f"  Density: {len(density_data['occupation_density'])} occupations")
    print(f"  AI exposure: {len(exposure_data['scores'])} occupations")

    # ================================================================
    # 2. Merge into flat records
    # ================================================================
    print("\n2. Merging into flat per-occupation records...")

    benchmarks = {}
    density_available_socs = []
    density_missing_socs = []

    for soc in matched_socs:
        record = {}

        # BLS projections (always available for matched SOCs)
        bls = bls_data["projections"][soc]
        record["title"] = bls["title"]
        record["employment_2024"] = bls["employment_2024"]
        record["growth_rate"] = bls["growth_rate"]
        record["median_annual_wage"] = bls["median_annual_wage"]
        record["wage_top_coded"] = bls["median_annual_wage"] == 239200 if bls["median_annual_wage"] is not None else False
        record["typical_education"] = bls["typical_education"]

        # Transferability (always available)
        trans = transfer_data["scores"][soc]
        record["transferability_score"] = trans["transferability_score"]
        record["transferability_percentile"] = trans["percentile"]
        # Z-score will be computed later
        record["transferability_z"] = None

        # Growing neighbors
        record["growing_neighbors"] = [
            {
                "soc": n["soc"],
                "title": n["title"],
                "similarity": n["similarity"],
                "growth_rate": n["growth_rate"],
                "employment_2024": n["employment_2024"],
            }
            for n in trans.get("growing_neighbors", [])
        ]

        # Density (may be missing for 11 occupations)
        if soc in density_data["occupation_density"]:
            dens = density_data["occupation_density"][soc]
            record["density_score"] = dens["density_score"]
            record["density_percentile"] = dens["percentile"]
            record["density_available"] = True
            record["density_low_data"] = dens["low_data"]
            record["density_z"] = None  # computed later
            density_available_socs.append(soc)
        else:
            record["density_score"] = None
            record["density_percentile"] = None
            record["density_available"] = False
            record["density_low_data"] = None
            record["density_z"] = None
            density_missing_socs.append(soc)

        # AI exposure (always available)
        exp = exposure_data["scores"][soc]
        record["ai_exposure"] = exp["exposure_score"]
        record["ai_exposure_human"] = exp["human_exposure_score"]
        record["ai_exposure_percentile"] = exp["percentile"]

        # Composite — computed in step 3
        record["composite_score"] = None
        record["composite_z"] = None

        benchmarks[soc] = record

    print(f"  Merged {len(benchmarks)} occupations")
    print(f"  Full data (with density): {len(density_available_socs)}")
    print(f"  Missing density: {len(density_missing_socs)}")

    # ================================================================
    # 3. Compute 2-component partial composite
    # ================================================================
    print("\n3. Computing 2-component partial composite...")

    # Work with the 740 occupations that have both components
    composite_socs = sorted(density_available_socs)
    N_comp = len(composite_socs)

    # Arrays for vectorized operations
    trans_vals = np.array([benchmarks[soc]["transferability_score"] for soc in composite_socs])
    dens_vals = np.array([benchmarks[soc]["density_score"] for soc in composite_socs])
    emp_vals = np.array([benchmarks[soc]["employment_2024"] for soc in composite_socs])

    # Step 3a: Winsorize at employment-weighted 1st and 99th percentiles
    print("\n  3a. Winsorizing at employment-weighted 1st/99th percentiles...")
    trans_wins, trans_p1, trans_p99 = winsorize_weighted(trans_vals, emp_vals, 1, 99)
    dens_wins, dens_p1, dens_p99 = winsorize_weighted(dens_vals, emp_vals, 1, 99)
    print(f"    Transferability: clipped to [{trans_p1:.6f}, {trans_p99:.6f}]")
    print(f"    Density: clipped to [{dens_p1:.6f}, {dens_p99:.6f}]")

    trans_clipped = np.sum(trans_vals != trans_wins)
    dens_clipped = np.sum(dens_vals != dens_wins)
    print(f"    Transferability values clipped: {trans_clipped}")
    print(f"    Density values clipped: {dens_clipped}")

    # Step 3b: Employment-weighted Z-scores
    print("\n  3b. Computing employment-weighted Z-scores...")
    trans_z, trans_wmean, trans_wstd = weighted_zscore(trans_wins, emp_vals)
    dens_z, dens_wmean, dens_wstd = weighted_zscore(dens_wins, emp_vals)
    print(f"    Transferability: wmean={trans_wmean:.6f}, wstd={trans_wstd:.6f}")
    print(f"    Density: wmean={dens_wmean:.6f}, wstd={dens_wstd:.6f}")

    # Step 3c: Average the 2 Z-scores (equal weight)
    print("\n  3c. Averaging Z-scores (0.5 transferability + 0.5 density)...")
    composite_z = 0.5 * trans_z + 0.5 * dens_z
    print(f"    Composite Z range: {composite_z.min():.4f} to {composite_z.max():.4f}")
    print(f"    Composite Z mean: {composite_z.mean():.6f}")

    # Step 3d: Employment-weighted percentile rank
    print("\n  3d. Computing employment-weighted percentile ranks...")
    composite_pctl = employment_weighted_percentile_rank(composite_z, emp_vals)
    print(f"    Composite percentile range: {composite_pctl.min():.2f} to {composite_pctl.max():.2f}")

    # Write Z-scores and composite back into records
    for i, soc in enumerate(composite_socs):
        benchmarks[soc]["transferability_z"] = round(float(trans_z[i]), 4)
        benchmarks[soc]["density_z"] = round(float(dens_z[i]), 4)
        benchmarks[soc]["composite_z"] = round(float(composite_z[i]), 4)
        benchmarks[soc]["composite_score"] = round(float(composite_pctl[i]), 2)

    print(f"\n  Computed composite for {N_comp} occupations")

    # ================================================================
    # 4. Sanity checks
    # ================================================================
    all_pass = run_sanity_checks(benchmarks, composite_socs, matched_socs)

    if not all_pass:
        print("\nWARNING: Some sanity checks failed. Review output carefully.")

    # ================================================================
    # 5. Build output JSON
    # ================================================================
    print("\n" + "=" * 60)
    print("WRITING OUTPUT")
    print("=" * 60)

    comp_arr = np.array([benchmarks[soc]["composite_score"] for soc in composite_socs])

    output = {
        "metadata": {
            "description": "Occupation-level benchmark table for adaptive capacity tool",
            "methodology": "Manning & Aguirre (2026), NBER Working Paper 34705",
            "composite_components": ["transferability", "density"],
            "composite_note": (
                "Partial composite from 2 of 4 study components "
                "(transferability + density). Wealth and age are user-inputted "
                "at runtime. Methodology: winsorize at employment-weighted "
                "1st/99th percentiles, compute employment-weighted Z-scores, "
                "average with equal weights, then employment-weighted "
                "percentile rank."
            ),
            "winsorization": {
                "transferability": {
                    "p1": round(float(trans_p1), 6),
                    "p99": round(float(trans_p99), 6),
                },
                "density": {
                    "p1": round(float(dens_p1), 6),
                    "p99": round(float(dens_p99), 6),
                },
            },
            "weighted_means": {
                "transferability": round(float(trans_wmean), 6),
                "density": round(float(dens_wmean), 6),
            },
            "weighted_stds": {
                "transferability": round(float(trans_wstd), 6),
                "density": round(float(dens_wstd), 6),
            },
            "total_occupations": len(matched_socs),
            "occupations_with_composite": N_comp,
            "occupations_missing_density": len(density_missing_socs),
            "composite_range": {
                "min": round(float(comp_arr.min()), 2),
                "max": round(float(comp_arr.max()), 2),
                "mean": round(float(comp_arr.mean()), 2),
                "median": round(float(np.median(comp_arr)), 2),
                "std": round(float(comp_arr.std()), 2),
            },
            "generated_by": "scripts/07_build_benchmarks.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "occupations": {},
    }

    for soc in matched_socs:
        output["occupations"][soc] = benchmarks[soc]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  Wrote {OUTPUT_FILE} ({file_size:.2f} MB)")

    # ================================================================
    # 6. Sample output
    # ================================================================
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    # Distribution stats
    print(f"\n  Composite distribution (N={N_comp}):")
    pcts = np.percentile(comp_arr, [5, 25, 50, 75, 95])
    print(f"    Min: {comp_arr.min():.2f}  Max: {comp_arr.max():.2f}")
    print(f"    5th: {pcts[0]:.2f}  25th: {pcts[1]:.2f}  50th: {pcts[2]:.2f}  75th: {pcts[3]:.2f}  95th: {pcts[4]:.2f}")
    print(f"    Mean: {comp_arr.mean():.2f}  Std: {comp_arr.std():.2f}")

    # Top 5 by composite
    ranked = sorted(
        [(soc, benchmarks[soc]) for soc in composite_socs],
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )

    print("\n  Top 5 by composite score:")
    for soc, v in ranked[:5]:
        print(
            f"    {v['composite_score']:6.2f}  {soc} — {v['title']}"
            f"  (T_z={v['transferability_z']:+.2f}, D_z={v['density_z']:+.2f})"
        )

    print("\n  Bottom 5 by composite score:")
    for soc, v in ranked[-5:]:
        print(
            f"    {v['composite_score']:6.2f}  {soc} — {v['title']}"
            f"  (T_z={v['transferability_z']:+.2f}, D_z={v['density_z']:+.2f})"
        )

    # Spotlight occupations
    spotlight = [
        ("15-1252", "Software Developers"),
        ("29-1141", "Registered Nurses"),
        ("11-1011", "Chief Executives"),
        ("11-1021", "General and Operations Managers"),
        ("47-2031", "Carpenters"),
    ]
    print("\n  Spotlight occupations:")
    for soc, name in spotlight:
        v = benchmarks.get(soc, {})
        comp = v.get("composite_score")
        comp_str = f"{comp:.2f}" if comp is not None else "N/A"
        print(
            f"    {soc} — {v.get('title', name)}: "
            f"composite={comp_str}, "
            f"T_pctl={v.get('transferability_percentile', 'N/A')}, "
            f"D_pctl={v.get('density_percentile', 'N/A')}, "
            f"AI_exp={v.get('ai_exposure', 'N/A')}"
        )

    # Density-missing occupations
    if density_missing_socs:
        print(f"\n  Occupations missing density ({len(density_missing_socs)}):")
        for soc in density_missing_socs:
            v = benchmarks[soc]
            print(f"    {soc} — {v['title']} (emp={v['employment_2024']}K)")

    print(f"\n  File size: {file_size:.2f} MB")
    print(f"  Total occupations: {len(matched_socs)}")
    print(f"  With composite: {N_comp}")


if __name__ == "__main__":
    main()
