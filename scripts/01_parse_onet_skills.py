#!/usr/bin/env python3
"""
Step 1: Parse O*NET 30.1 Skills and Work Activities into skill profile vectors.

Downloads and parses the O*NET 30.1 database to extract:
- 35 Skill importance ratings per occupation
- 41 Work Activity importance ratings per occupation
= 76-dimension skill vector per occupation

Outputs:
- data/onet_skill_profiles.json: SOC code -> 76-dimension skill vector
- Also builds the SOC code mapping (8-digit O*NET -> 6-digit BLS)

Data source: O*NET 30.1 (December 2025)
https://www.onetcenter.org/database.html
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
ONET_DIR = Path("/tmp/onet_30_1/db_30_1_text")

SKILLS_FILE = ONET_DIR / "Skills.txt"
WORK_ACTIVITIES_FILE = ONET_DIR / "Work Activities.txt"
OCCUPATION_DATA_FILE = ONET_DIR / "Occupation Data.txt"

OUTPUT_FILE = DATA_DIR / "onet_skill_profiles.json"
SOC_MAPPING_FILE = DATA_DIR / "soc_code_mapping.json"


def load_occupation_titles():
    """Load occupation titles from Occupation Data.txt."""
    titles = {}
    with open(OCCUPATION_DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            titles[row["O*NET-SOC Code"]] = row["Title"]
    return titles


def parse_importance_ratings(filepath, element_type):
    """
    Parse importance (IM scale) ratings from an O*NET data file.

    Returns: dict of {soc_code: {element_name: importance_value}}
    """
    ratings = defaultdict(dict)
    element_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Only use Importance (IM) scale, not Level (LV)
            if row["Scale ID"] != "IM":
                continue

            # Include all values, even suppressed ones.
            # Suppressed values (e.g., IM=1.0 for "Repairing Mechanical Equipment"
            # for Writers) are valid low-importance ratings needed for cosine
            # similarity computation.

            soc_code = row["O*NET-SOC Code"]
            element_name = row["Element Name"]
            element_id = row["Element ID"]
            value = float(row["Data Value"])

            ratings[soc_code][element_id] = {
                "name": element_name,
                "value": value,
                "type": element_type,
            }
            element_ids.add((element_id, element_name))

    return ratings, sorted(element_ids)


def build_soc_mapping(onet_codes):
    """
    Build mapping between O*NET 8-digit SOC codes and BLS 6-digit SOC codes.

    O*NET uses codes like 15-1252.00 (standard) or 15-1252.01 (detailed).
    BLS uses 6-digit codes like 15-1252.

    For .00 codes: direct 1:1 mapping to BLS 6-digit.
    For .XX codes (specialties): map to the parent .00 / BLS 6-digit code.
    """
    mapping = {}
    for code in onet_codes:
        bls_code = code.split(".")[0]  # e.g., "15-1252.00" -> "15-1252"
        suffix = code.split(".")[1]  # e.g., "00" or "01"
        mapping[code] = {
            "bls_soc": bls_code,
            "is_detailed": suffix != "00",
            "suffix": suffix,
        }
    return mapping


def aggregate_to_bls_soc(skill_ratings, work_activity_ratings, element_order):
    """
    Aggregate O*NET detailed occupation codes to BLS 6-digit SOC level.

    Strategy:
    - .00 codes map directly
    - .XX specialty codes: if a .00 parent exists, skip the specialty
      (the .00 already represents the aggregate)
    - .XX specialty codes without a .00 parent: average the specialties

    This ensures we have one profile per BLS 6-digit code.
    """
    # Find all O*NET codes that have skill data
    all_codes = set(skill_ratings.keys()) & set(work_activity_ratings.keys())

    # Group by BLS 6-digit code
    bls_groups = defaultdict(list)
    for code in all_codes:
        bls_code = code.split(".")[0]
        bls_groups[bls_code].append(code)

    profiles = {}
    onet_to_bls = {}

    for bls_code, onet_codes in sorted(bls_groups.items()):
        # Prefer .00 if it exists
        standard_code = f"{bls_code}.00"
        if standard_code in onet_codes:
            source_codes = [standard_code]
        else:
            # No .00 parent — use all specialty codes and average
            source_codes = sorted(onet_codes)

        # Build the 76-dimension vector by averaging across source codes
        vector = {}
        for element_id, element_name in element_order:
            values = []
            for code in source_codes:
                if code in skill_ratings and element_id in skill_ratings[code]:
                    values.append(skill_ratings[code][element_id]["value"])
                elif (
                    code in work_activity_ratings
                    and element_id in work_activity_ratings[code]
                ):
                    values.append(work_activity_ratings[code][element_id]["value"])

            if values:
                vector[element_id] = round(sum(values) / len(values), 4)
            else:
                vector[element_id] = None

        profiles[bls_code] = {
            "vector": vector,
            "source_onet_codes": source_codes,
        }

        # Track mapping
        for code in onet_codes:
            onet_to_bls[code] = bls_code

    return profiles, onet_to_bls


def main():
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Verify input files exist
    for f in [SKILLS_FILE, WORK_ACTIVITIES_FILE, OCCUPATION_DATA_FILE]:
        if not f.exists():
            print(f"ERROR: Required file not found: {f}")
            print("Run the O*NET download first.")
            sys.exit(1)

    print("Loading occupation titles...")
    titles = load_occupation_titles()
    print(f"  Found {len(titles)} occupation entries in Occupation Data")

    print("\nParsing Skills importance ratings...")
    skill_ratings, skill_elements = parse_importance_ratings(SKILLS_FILE, "skill")
    print(f"  Found {len(skill_elements)} skill elements across {len(skill_ratings)} occupations")

    print("\nParsing Work Activities importance ratings...")
    wa_ratings, wa_elements = parse_importance_ratings(WORK_ACTIVITIES_FILE, "work_activity")
    print(f"  Found {len(wa_elements)} work activity elements across {len(wa_ratings)} occupations")

    total_elements = len(skill_elements) + len(wa_elements)
    print(f"\nTotal dimensions: {total_elements} ({len(skill_elements)} skills + {len(wa_elements)} work activities)")

    # Combined element order (skills first, then work activities)
    element_order = skill_elements + wa_elements

    # Build element reference
    element_reference = []
    for eid, ename in element_order:
        etype = "skill" if eid.startswith("2.") else "work_activity"
        element_reference.append({
            "element_id": eid,
            "element_name": ename,
            "type": etype,
        })

    print("\nAggregating to BLS 6-digit SOC codes...")
    profiles, onet_to_bls = aggregate_to_bls_soc(
        skill_ratings, wa_ratings, element_order
    )
    print(f"  Produced {len(profiles)} BLS-level occupation profiles")

    # Attach titles to profiles
    for bls_code, profile in profiles.items():
        # Get title from the source O*NET code(s)
        source_titles = []
        for onet_code in profile["source_onet_codes"]:
            if onet_code in titles:
                source_titles.append(titles[onet_code])
        profile["title"] = source_titles[0] if source_titles else bls_code

        # Get all related O*NET titles for reference
        all_related = []
        for onet_code in sorted(
            [c for c in titles if c.startswith(bls_code + ".")]
        ):
            all_related.append({"onet_code": onet_code, "title": titles[onet_code]})
        profile["related_onet_occupations"] = all_related

    # Check for any profiles with missing values
    complete = 0
    incomplete = 0
    for bls_code, profile in profiles.items():
        if None in profile["vector"].values():
            incomplete += 1
        else:
            complete += 1
    print(f"  Complete profiles (all 76 values): {complete}")
    print(f"  Incomplete profiles (missing values): {incomplete}")

    # Build output
    output = {
        "metadata": {
            "source": "O*NET 30.1 Database (December 2025)",
            "url": "https://www.onetcenter.org/database.html",
            "description": "Skill profiles with 76-dimension importance vectors (35 skills + 41 work activities) per occupation",
            "scale": "Importance (IM) - range 1-5",
            "total_occupations": len(profiles),
            "dimensions": total_elements,
        },
        "elements": element_reference,
        "profiles": {},
    }

    for bls_code in sorted(profiles.keys()):
        p = profiles[bls_code]
        output["profiles"][bls_code] = {
            "title": p["title"],
            "vector": p["vector"],
            "source_onet_codes": p["source_onet_codes"],
            "related_onet_occupations": p["related_onet_occupations"],
        }

    # Write skill profiles JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nWrote {OUTPUT_FILE} ({file_size:.1f} MB)")

    # Build and write SOC code mapping
    all_onet_codes = set()
    for p in profiles.values():
        all_onet_codes.update(
            o["onet_code"] for o in p["related_onet_occupations"]
        )
    soc_mapping = build_soc_mapping(sorted(all_onet_codes))

    soc_mapping_output = {
        "metadata": {
            "description": "Mapping between O*NET 8-digit SOC codes and BLS 6-digit SOC codes",
            "total_onet_codes": len(soc_mapping),
            "total_bls_codes": len(profiles),
        },
        "mapping": soc_mapping,
    }

    with open(SOC_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(soc_mapping_output, f, indent=2)
    print(f"Wrote {SOC_MAPPING_FILE}")

    # Print sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (first 3 occupations)")
    print("=" * 70)
    sample_codes = sorted(profiles.keys())[:3]
    for bls_code in sample_codes:
        p = output["profiles"][bls_code]
        print(f"\n{bls_code}: {p['title']}")
        print(f"  Source O*NET codes: {p['source_onet_codes']}")
        print(f"  Related occupations: {len(p['related_onet_occupations'])}")
        # Show first 5 skill values and first 5 work activity values
        skills_shown = 0
        wa_shown = 0
        print("  Skills (first 5):")
        for eid, val in p["vector"].items():
            if eid.startswith("2.") and skills_shown < 5:
                ename = next(
                    e["element_name"]
                    for e in element_reference
                    if e["element_id"] == eid
                )
                print(f"    {ename}: {val}")
                skills_shown += 1
        print("  Work Activities (first 5):")
        for eid, val in p["vector"].items():
            if eid.startswith("4.") and wa_shown < 5:
                ename = next(
                    e["element_name"]
                    for e in element_reference
                    if e["element_id"] == eid
                )
                print(f"    {ename}: {val}")
                wa_shown += 1


if __name__ == "__main__":
    main()
