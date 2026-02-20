#!/usr/bin/env python3
"""
Step 6: Bundle AI exposure scores from Eloundou et al. (2024).

Downloads occupation-level AI exposure scores from the GPTs-are-GPTs GitHub
repository (Eloundou, Manning, Mishkin & Rock, 2024). The primary score is the
"beta" measure: E1 + 0.5×E2, representing the proportion of an occupation's
tasks exposed to LLMs (direct exposure at full weight, tool-augmented exposure
at half weight).

Data source:
  Eloundou et al. (2024). "GPTs are GPTs: Labor Market Impact Potential of LLMs."
  Science 384(6702):1306-1308. DOI: 10.1126/science.adj0998
  GitHub: https://github.com/openai/GPTs-are-GPTs

Inputs:
  - data/transferability_scores.json (from Step 4, for 751 matched SOC codes)

Outputs:
  - data/ai_exposure.json

Based on Manning & Aguirre (2026), "How Adaptable Are American Workers
to AI-Induced Job Displacement?", NBER Working Paper 34705.
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

TRANSFER_FILE = DATA_DIR / "transferability_scores.json"
OUTPUT_FILE = DATA_DIR / "ai_exposure.json"

# Source data
ELOUNDOU_URL = "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/occ_level.csv"
CACHE_PATH = Path("/tmp/eloundou_occ_level.csv")

# Expected CSV columns
EXPECTED_COLUMNS = [
    "O*NET-SOC Code",
    "Title",
    "dv_rating_alpha",
    "dv_rating_beta",
    "dv_rating_gamma",
    "human_rating_alpha",
    "human_rating_beta",
    "human_rating_gamma",
]

SCORE_COLUMNS = [
    "dv_rating_alpha",
    "dv_rating_beta",
    "dv_rating_gamma",
    "human_rating_alpha",
    "human_rating_beta",
    "human_rating_gamma",
]


def download_csv(url, local_path, min_size=5000):
    """
    Download the Eloundou et al. CSV from GitHub.

    GitHub raw URLs don't have bot detection, so a simple curl works.
    Validates that the downloaded file is actually a CSV (not an HTML error page).
    """
    if local_path.exists():
        size = local_path.stat().st_size
        if size > min_size:
            # Quick sanity check: first line should look like CSV headers
            with open(local_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if "O*NET-SOC Code" in first_line:
                print(f"  Using cached file: {local_path} ({size / 1024:.0f} KB)")
                return
            else:
                print(f"  Cached file is invalid (not CSV), re-downloading...")
                local_path.unlink()

    print(f"  Downloading from {url}...")
    result = subprocess.run(
        ["curl", "-L", "-o", str(local_path), url],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR: curl failed with return code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        print(f"\n  Manual download: visit {url}")
        print(f"  Save to: {local_path}")
        sys.exit(1)

    # Validate it's a real CSV
    if not local_path.exists():
        print(f"  ERROR: File not created: {local_path}")
        sys.exit(1)

    size = local_path.stat().st_size
    if size < min_size:
        print(f"  ERROR: File too small ({size} bytes). May be an error page.")
        local_path.unlink()
        sys.exit(1)

    with open(local_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if "O*NET-SOC Code" not in first_line:
        print(f"  ERROR: File doesn't look like the expected CSV.")
        print(f"  First line: {first_line[:200]}")
        local_path.unlink()
        sys.exit(1)

    print(f"  Downloaded: {local_path} ({size / 1024:.0f} KB)")


def parse_csv(filepath):
    """
    Parse the Eloundou et al. occupation-level CSV.

    Returns:
        rows: list of dicts, each with onet_code, title, and all 6 score fields
    """
    rows = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate columns
        actual_cols = reader.fieldnames
        missing = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
        if missing:
            print(f"  ERROR: Missing expected columns: {missing}")
            print(f"  Found columns: {actual_cols}")
            sys.exit(1)

        for row in reader:
            onet_code = row["O*NET-SOC Code"].strip()
            title = row["Title"].strip()

            # Parse all 6 score columns
            scores = {}
            valid = True
            for col in SCORE_COLUMNS:
                try:
                    scores[col] = float(row[col])
                except (ValueError, TypeError):
                    print(f"  WARNING: Non-numeric score for {onet_code} / {col}: {row[col]}")
                    valid = False
                    break

            if not valid:
                continue

            rows.append({
                "onet_code": onet_code,
                "title": title,
                "scores": scores,
            })

    return rows


def aggregate_to_bls(rows):
    """
    Map O*NET 8-digit codes to BLS 6-digit codes.

    Strategy (same as Step 1):
    - Group rows by BLS 6-digit code (strip .XX suffix)
    - If a .00 code exists → use only that row (canonical occupation)
    - If only specialty codes (.01, .02, ...) → average their scores
    """
    # Group by BLS code
    bls_groups = defaultdict(list)
    for row in rows:
        onet_code = row["onet_code"]
        bls_code = onet_code.split(".")[0]
        suffix = onet_code.split(".")[1] if "." in onet_code else "00"
        bls_groups[bls_code].append({
            "suffix": suffix,
            "onet_code": onet_code,
            "title": row["title"],
            "scores": row["scores"],
        })

    # Aggregate
    bls_scores = {}
    for bls_code, entries in sorted(bls_groups.items()):
        # Prefer .00 if it exists
        standard = [e for e in entries if e["suffix"] == "00"]
        if standard:
            source_entries = standard
            aggregation = "direct"
        else:
            source_entries = entries
            aggregation = "averaged"

        # Average scores across source entries
        avg_scores = {}
        for col in SCORE_COLUMNS:
            values = [e["scores"][col] for e in source_entries]
            avg_scores[col] = sum(values) / len(values)

        # Use title from .00 or first entry
        title = source_entries[0]["title"]

        bls_scores[bls_code] = {
            "title": title,
            "scores": avg_scores,
            "source_onet_codes": [e["onet_code"] for e in entries],
            "aggregation": aggregation,
        }

    return bls_scores


def load_pipeline_occupations(filepath):
    """Load the 751 matched SOC codes from Step 4 output."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = sorted(data["scores"].keys())
    titles = {soc: v["title"] for soc, v in data["scores"].items()}
    return codes, titles


def compute_ranks_percentiles(scores_dict):
    """
    Compute ranks and percentiles for exposure scores.

    Rank 1 = highest exposure. Percentile 100 = most exposed.
    """
    soc_codes = sorted(scores_dict.keys())
    values = np.array([scores_dict[soc]["scores"]["dv_rating_beta"] for soc in soc_codes])

    # Rank: 1 = highest score (most exposed)
    order = np.argsort(-values)  # descending
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)

    N = len(values)
    for i, soc in enumerate(soc_codes):
        scores_dict[soc]["rank"] = int(ranks[i])
        scores_dict[soc]["percentile"] = round((N - ranks[i]) / (N - 1) * 100, 2)


def run_sanity_checks(bls_scores, matched_scores, pipeline_codes, unmatched_ours):
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

    # 1. CSV row count
    check(
        "CSV has ≥800 O*NET occupations",
        len(bls_scores) >= 400,  # After BLS aggregation, ~800 O*NET → ~800 BLS
        f"{len(bls_scores)} BLS-level occupations from CSV",
    )

    # 2. Score range
    all_betas = [v["scores"]["dv_rating_beta"] for v in matched_scores.values()]
    min_beta = min(all_betas)
    max_beta = max(all_betas)
    check(
        "All dv_beta scores in [0.0, 1.0]",
        min_beta >= 0.0 and max_beta <= 1.0,
        f"range: {min_beta:.4f} to {max_beta:.4f}",
    )

    all_human_betas = [v["scores"]["human_rating_beta"] for v in matched_scores.values()]
    min_hb = min(all_human_betas)
    max_hb = max(all_human_betas)
    check(
        "All human_beta scores in [0.0, 1.0]",
        min_hb >= 0.0 and max_hb <= 1.0,
        f"range: {min_hb:.4f} to {max_hb:.4f}",
    )

    # 3. No NaN
    check(
        "No NaN in dv_beta scores",
        not any(np.isnan(v) for v in all_betas),
    )

    # 4. Coverage
    N = len(matched_scores)
    check(
        f"Coverage ≥ 680 of {len(pipeline_codes)} pipeline occupations",
        N >= 680,
        f"{N} scored ({N / len(pipeline_codes) * 100:.1f}%)",
    )

    # 5. Alpha ≤ beta ≤ gamma monotonicity
    mono_violations = 0
    for soc, v in matched_scores.items():
        s = v["scores"]
        # Check for both GPT-4 and human ratings
        if s["dv_rating_alpha"] > s["dv_rating_beta"] + 1e-6:
            mono_violations += 1
        if s["dv_rating_beta"] > s["dv_rating_gamma"] + 1e-6:
            mono_violations += 1
        if s["human_rating_alpha"] > s["human_rating_beta"] + 1e-6:
            mono_violations += 1
        if s["human_rating_beta"] > s["human_rating_gamma"] + 1e-6:
            mono_violations += 1
    check(
        "Alpha ≤ beta ≤ gamma for all occupations (both raters)",
        mono_violations == 0,
        f"{mono_violations} violations" if mono_violations > 0 else "all monotonic",
    )

    # 6. Face validity: Data Entry Keyers (43-9021) should be high exposure
    if "43-9021" in matched_scores:
        dek_pct = matched_scores["43-9021"]["percentile"]
        check(
            "Data Entry Keyers (43-9021) in top quartile",
            dek_pct >= 75.0,
            f"percentile: {dek_pct:.1f}",
        )
    else:
        print("  [SKIP] Data Entry Keyers (43-9021) — not in matched set")

    # 7. Face validity: physical occupation should be low exposure
    # Try Athletes and Sports Competitors, then Roofers, then Loggers
    physical_checks = [
        ("27-2021", "Athletes and Sports Competitors"),
        ("47-2181", "Roofers"),
        ("45-4022", "Logging Equipment Operators"),
        ("47-2031", "Carpenters"),
    ]
    found_physical = False
    for soc, name in physical_checks:
        if soc in matched_scores:
            phys_pct = matched_scores[soc]["percentile"]
            check(
                f"{name} ({soc}) in bottom quartile",
                phys_pct <= 25.0,
                f"percentile: {phys_pct:.1f}",
            )
            found_physical = True
            break
    if not found_physical:
        print("  [SKIP] Physical occupation check — none found in matched set")

    # 8. Face validity: Software Developers above median
    if "15-1252" in matched_scores:
        dev_pct = matched_scores["15-1252"]["percentile"]
        check(
            "Software Developers (15-1252) above median",
            dev_pct >= 50.0,
            f"percentile: {dev_pct:.1f}",
        )
    else:
        print("  [SKIP] Software Developers (15-1252) — not in matched set")

    # 9. Correlation between GPT-4 and human ratings
    dv_betas = np.array(all_betas)
    h_betas = np.array(all_human_betas)
    correlation = np.corrcoef(dv_betas, h_betas)[0, 1]
    check(
        "GPT-4 / human beta correlation > 0.7",
        correlation > 0.7,
        f"r = {correlation:.4f}",
    )

    # 10. Distribution stats
    arr = np.array(all_betas)
    pcts = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"\n  Distribution (dv_rating_beta, N={N}):")
    print(f"    Min: {arr.min():.4f}  Max: {arr.max():.4f}")
    print(f"    5th: {pcts[0]:.4f}  25th: {pcts[1]:.4f}  50th: {pcts[2]:.4f}  75th: {pcts[3]:.4f}  95th: {pcts[4]:.4f}")
    print(f"    Mean: {arr.mean():.4f}  Std: {arr.std():.4f}")

    h_arr = np.array(all_human_betas)
    h_pcts = np.percentile(h_arr, [5, 25, 50, 75, 95])
    print(f"\n  Distribution (human_rating_beta, N={N}):")
    print(f"    Min: {h_arr.min():.4f}  Max: {h_arr.max():.4f}")
    print(f"    5th: {h_pcts[0]:.4f}  25th: {h_pcts[1]:.4f}  50th: {h_pcts[2]:.4f}  75th: {h_pcts[3]:.4f}  95th: {h_pcts[4]:.4f}")
    print(f"    Mean: {h_arr.mean():.4f}  Std: {h_arr.std():.4f}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def main():
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Verify input exists
    if not TRANSFER_FILE.exists():
        print(f"ERROR: Input file not found: {TRANSFER_FILE}")
        print("Run scripts/04_compute_transferability.py first.")
        sys.exit(1)

    print("Step 6: Bundle AI Exposure Scores")
    print("=" * 60)

    # 1. Download CSV
    print("\n1. Downloading Eloundou et al. occupation-level scores...")
    download_csv(ELOUNDOU_URL, CACHE_PATH)

    # 2. Parse CSV
    print("\n2. Parsing CSV...")
    rows = parse_csv(CACHE_PATH)
    print(f"  Parsed {len(rows)} valid O*NET-level occupation rows")

    # 3. Aggregate to BLS 6-digit codes
    print("\n3. Mapping O*NET 8-digit → BLS 6-digit codes...")
    bls_scores = aggregate_to_bls(rows)
    direct_count = sum(1 for v in bls_scores.values() if v["aggregation"] == "direct")
    averaged_count = sum(1 for v in bls_scores.values() if v["aggregation"] == "averaged")
    print(f"  {len(bls_scores)} BLS-level occupations")
    print(f"    Direct (.00 code): {direct_count}")
    print(f"    Averaged (specialty codes): {averaged_count}")

    # 4. Load pipeline occupations
    print("\n4. Loading pipeline occupations from Step 4...")
    pipeline_codes, pipeline_titles = load_pipeline_occupations(TRANSFER_FILE)
    print(f"  {len(pipeline_codes)} pipeline occupations")

    # 5. Join datasets
    # Fallback mapping for occupations in our pipeline but missing from
    # the Eloundou source (e.g., 13-1082 Project Management Specialists
    # is not in the O*NET-era Eloundou CSV). Use a related occupation's
    # scores as a proxy.
    AI_EXPOSURE_FALLBACKS = {
        "13-1082": "13-1111",  # Project Management Specialists → Management Analysts
    }

    print("\n5. Matching to pipeline occupations...")
    matched_scores = {}
    unmatched_ours = []
    unmatched_eloundou = []
    fallback_used = []

    for soc in pipeline_codes:
        if soc in bls_scores:
            entry = bls_scores[soc].copy()
            # Use pipeline title (from BLS projections) as primary
            entry["title"] = pipeline_titles[soc]
            matched_scores[soc] = entry
        elif soc in AI_EXPOSURE_FALLBACKS:
            proxy_soc = AI_EXPOSURE_FALLBACKS[soc]
            if proxy_soc in bls_scores:
                entry = bls_scores[proxy_soc].copy()
                entry["title"] = pipeline_titles[soc]
                entry["aggregation"] = f"proxy_from_{proxy_soc}"
                entry["source_onet_codes"] = bls_scores[proxy_soc]["source_onet_codes"]
                matched_scores[soc] = entry
                fallback_used.append((soc, proxy_soc))
            else:
                unmatched_ours.append(soc)
        else:
            unmatched_ours.append(soc)

    for soc in bls_scores:
        if soc not in pipeline_codes:
            unmatched_eloundou.append(soc)

    if fallback_used:
        print(f"  Fallback proxy scores used for {len(fallback_used)} occupations:")
        for soc, proxy in fallback_used:
            print(f"    {soc} ({pipeline_titles[soc]}) ← {proxy} ({pipeline_titles.get(proxy, '?')})")

    print(f"  Matched: {len(matched_scores)} occupations")
    print(f"  Pipeline occupations without Eloundou scores: {len(unmatched_ours)}")
    print(f"  Eloundou occupations not in pipeline: {len(unmatched_eloundou)}")

    if unmatched_ours:
        print(f"\n  Unmatched pipeline occupations (first 10):")
        for soc in unmatched_ours[:10]:
            print(f"    {soc}: {pipeline_titles[soc]}")
        if len(unmatched_ours) > 10:
            print(f"    ... and {len(unmatched_ours) - 10} more")

    # 6. Compute ranks and percentiles
    print("\n6. Computing ranks and percentiles...")
    compute_ranks_percentiles(matched_scores)
    print(f"  Ranked {len(matched_scores)} occupations")

    # 7. Sanity checks
    all_pass = run_sanity_checks(
        bls_scores, matched_scores, pipeline_codes, unmatched_ours
    )

    if not all_pass:
        print("\nWARNING: Some sanity checks failed. Review output carefully.")

    # 8. Build output JSON
    print("\n" + "=" * 60)
    print("WRITING OUTPUT")
    print("=" * 60)

    all_betas = np.array([v["scores"]["dv_rating_beta"] for v in matched_scores.values()])
    all_human_betas = np.array([v["scores"]["human_rating_beta"] for v in matched_scores.values()])

    output = {
        "metadata": {
            "description": "AI exposure scores per occupation from Eloundou et al. (2024)",
            "source_paper": (
                "Eloundou, Manning, Mishkin & Rock (2024). "
                "GPTs are GPTs: Labor Market Impact Potential of LLMs. "
                "Science 384(6702):1306-1308. DOI: 10.1126/science.adj0998"
            ),
            "source_data": "https://github.com/openai/GPTs-are-GPTs/blob/main/data/occ_level.csv",
            "primary_score": "dv_rating_beta (GPT-4-rated E1+0.5E2)",
            "alternative_score": "human_rating_beta (Human-rated E1+0.5E2)",
            "score_interpretation": (
                "Proportion of occupation tasks exposed to LLMs. "
                "E1 = direct LLM exposure (full weight), "
                "E2 = tool-augmented LLM exposure (0.5 weight). "
                "Range 0.0 (no tasks exposed) to 1.0 (all tasks fully exposed)."
            ),
            "soc_aggregation": (
                "O*NET 8-digit codes mapped to BLS 6-digit. "
                "Prefer .00 (canonical); average specialty codes if no .00 exists."
            ),
            "total_eloundou_onet_rows": len(rows),
            "total_bls_mapped": len(bls_scores),
            "total_scored_in_pipeline": len(matched_scores),
            "unscored_pipeline_occupations": len(unmatched_ours),
            "score_range": {
                "dv_beta": {
                    "min": round(float(all_betas.min()), 4),
                    "max": round(float(all_betas.max()), 4),
                    "mean": round(float(all_betas.mean()), 4),
                    "median": round(float(np.median(all_betas)), 4),
                    "std": round(float(all_betas.std()), 4),
                },
                "human_beta": {
                    "min": round(float(all_human_betas.min()), 4),
                    "max": round(float(all_human_betas.max()), 4),
                    "mean": round(float(all_human_betas.mean()), 4),
                    "median": round(float(np.median(all_human_betas)), 4),
                    "std": round(float(all_human_betas.std()), 4),
                },
            },
            "gpt4_human_correlation": round(
                float(np.corrcoef(all_betas, all_human_betas)[0, 1]), 4
            ),
            "generated_by": "scripts/06_bundle_ai_exposure.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "scores": {},
    }

    for soc in sorted(matched_scores.keys()):
        v = matched_scores[soc]
        s = v["scores"]
        output["scores"][soc] = {
            "title": v["title"],
            "exposure_score": round(s["dv_rating_beta"], 4),
            "human_exposure_score": round(s["human_rating_beta"], 4),
            "rank": v["rank"],
            "percentile": v["percentile"],
            "all_scores": {
                "dv_alpha": round(s["dv_rating_alpha"], 4),
                "dv_beta": round(s["dv_rating_beta"], 4),
                "dv_gamma": round(s["dv_rating_gamma"], 4),
                "human_alpha": round(s["human_rating_alpha"], 4),
                "human_beta": round(s["human_rating_beta"], 4),
                "human_gamma": round(s["human_rating_gamma"], 4),
            },
            "source_onet_codes": v["source_onet_codes"],
            "aggregation": v["aggregation"],
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  Wrote {OUTPUT_FILE} ({file_size:.2f} MB)")

    # 9. Sample output
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    # Top 10 most exposed
    by_rank = sorted(matched_scores.items(), key=lambda x: x[1]["rank"])
    print("\n  Top 10 most AI-exposed occupations:")
    for soc, v in by_rank[:10]:
        print(
            f"    {v['rank']:3d}. {soc} — {v['title']}"
            f"  (dv_β={v['scores']['dv_rating_beta']:.3f}"
            f"  h_β={v['scores']['human_rating_beta']:.3f})"
        )

    # Bottom 10 least exposed
    print("\n  Bottom 10 least AI-exposed occupations:")
    for soc, v in by_rank[-10:]:
        print(
            f"    {v['rank']:3d}. {soc} — {v['title']}"
            f"  (dv_β={v['scores']['dv_rating_beta']:.3f}"
            f"  h_β={v['scores']['human_rating_beta']:.3f})"
        )

    # Median occupation
    median_idx = len(by_rank) // 2
    med_soc, med_v = by_rank[median_idx]
    print(f"\n  Median occupation:")
    print(
        f"    {med_v['rank']:3d}. {med_soc} — {med_v['title']}"
        f"  (dv_β={med_v['scores']['dv_rating_beta']:.3f}"
        f"  h_β={med_v['scores']['human_rating_beta']:.3f})"
    )

    # Specific occupations of interest
    spotlight = [
        ("15-1252", "Software Developers"),
        ("29-1141", "Registered Nurses"),
        ("11-1011", "Chief Executives"),
        ("43-9021", "Data Entry Keyers"),
        ("47-2031", "Carpenters"),
    ]
    print("\n  Spotlight occupations:")
    for soc, name in spotlight:
        if soc in matched_scores:
            v = matched_scores[soc]
            print(
                f"    {soc} — {v['title']}: "
                f"rank {v['rank']}/{len(matched_scores)}, "
                f"pctl {v['percentile']:.1f}, "
                f"dv_β={v['scores']['dv_rating_beta']:.3f}, "
                f"h_β={v['scores']['human_rating_beta']:.3f}, "
                f"src={v['source_onet_codes']}"
            )
        else:
            print(f"    {soc} — {name}: NOT IN MATCHED SET")

    # Coverage summary
    print(f"\n  Coverage: {len(matched_scores)}/{len(pipeline_codes)} pipeline occupations "
          f"({len(matched_scores) / len(pipeline_codes) * 100:.1f}%)")
    if unmatched_ours:
        print(f"  {len(unmatched_ours)} unscored occupations will need fallback values in Step 7")


if __name__ == "__main__":
    main()
