#!/usr/bin/env python3
"""
Step 8: Build quiz-filter decision tree for occupation selection.

Creates a JSON decision tree that narrows 751 occupations to ~5-15
candidates in 3-4 questions. The front-end traverses the tree to help
users find their occupation without scrolling a giant dropdown.

Strategy:
  Q1: Work setting (4 options — SOC major groups clustered by environment)
  Q2: Domain/field (varies by Q1 — SOC major group sub-categories)
  Q3: Education level (4 options — O*NET Job Zones mapped to education)
  Q4: Specialization (conditional — only if leaf > 15 occupations)

Inputs:
  - data/occupation_benchmarks.json (751 SOC codes + titles)
  - /tmp/onet_30_1/db_30_1_text/Job Zones.txt (O*NET Job Zones)
  - /tmp/onet_30_1/db_30_1_text/Occupation Data.txt (descriptions)

Outputs:
  - data/occupation_filter_tree.json
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

BENCHMARKS_FILE = DATA_DIR / "occupation_benchmarks.json"
JOB_ZONES_FILE = Path("/tmp/onet_30_1/db_30_1_text/Job Zones.txt")
OCC_DATA_FILE = Path("/tmp/onet_30_1/db_30_1_text/Occupation Data.txt")

OUTPUT_FILE = DATA_DIR / "occupation_filter_tree.json"

# Maximum leaf size before triggering Q4 split
MAX_LEAF_SIZE = 15

# ================================================================
# Tree structure definitions
# ================================================================

# Q1: Work setting → SOC major group mapping
Q1_BRANCHES = {
    "office": {
        "label": "Office, desk-based, or knowledge work",
        "soc_groups": ["11", "13", "15", "17", "19", "21", "23", "25", "27", "41", "43"],
    },
    "healthcare": {
        "label": "Healthcare or clinical settings",
        "soc_groups": ["29", "31"],
    },
    "trades": {
        "label": "Trades, construction, manufacturing, or transportation",
        "soc_groups": ["33", "37", "45", "47", "49", "51", "53"],
    },
    "service": {
        "label": "Food service or personal care",
        "soc_groups": ["35", "39"],
    },
}

# Q2: Domain/field → SOC major group sub-categories (per Q1 branch)
Q2_BRANCHES = {
    "office": {
        "text": "What is your primary field or function?",
        "options": {
            "tech_science": {
                "label": "Technology, science, or engineering",
                "soc_groups": ["15", "17", "19"],
            },
            "management": {
                "label": "Management or executive leadership",
                "soc_groups": ["11"],
            },
            "business_legal": {
                "label": "Business, finance, or legal",
                "soc_groups": ["13", "23"],
            },
            "education_social": {
                "label": "Education, social services, or community",
                "soc_groups": ["21", "25"],
            },
            "admin": {
                "label": "Office or administrative support",
                "soc_groups": ["43"],
            },
            "creative_sales": {
                "label": "Creative, media, or sales",
                "soc_groups": ["27", "41"],
            },
        },
    },
    "healthcare": {
        "text": "Which area of healthcare?",
        "options": {
            "clinical": {
                "label": "Clinical practice (doctors, nurses, therapists, technicians)",
                "soc_groups": ["29"],
            },
            "support": {
                "label": "Healthcare support (aides, assistants, attendants)",
                "soc_groups": ["31"],
            },
        },
    },
    "trades": {
        "text": "What type of work do you do?",
        "options": {
            "construction": {
                "label": "Construction or extraction",
                "soc_groups": ["47"],
            },
            "install_repair": {
                "label": "Installation, maintenance, or repair",
                "soc_groups": ["49"],
            },
            "production": {
                "label": "Manufacturing or production",
                "soc_groups": ["51"],
            },
            "transport": {
                "label": "Transportation, warehousing, or material moving",
                "soc_groups": ["53"],
            },
            "protective": {
                "label": "Protective services (police, fire, security)",
                "soc_groups": ["33"],
            },
            "farming": {
                "label": "Farming, fishing, forestry, or grounds maintenance",
                "soc_groups": ["37", "45"],
            },
        },
    },
    "service": {
        "text": "Which type of service work?",
        "options": {
            "food": {
                "label": "Food preparation or serving",
                "soc_groups": ["35"],
            },
            "personal": {
                "label": "Personal care or other services",
                "soc_groups": ["39"],
            },
        },
    },
}

# Q3: Education/Job Zone mapping
Q3_OPTIONS = {
    "hs_or_less": {
        "label": "High school diploma or less",
        "zones": [1, 2],
    },
    "vocational": {
        "label": "Vocational training, certificate, or associate degree",
        "zones": [3],
    },
    "bachelors": {
        "label": "Bachelor's degree",
        "zones": [4],
    },
    "graduate": {
        "label": "Graduate or professional degree",
        "zones": [5],
    },
}

# Human-readable labels for 4-digit SOC sub-groups (for Q4)
# These are hand-curated for the sub-groups that commonly appear
# in oversized leaves. The script falls back to the SOC group name
# if a label isn't defined here.
SOC_4DIGIT_LABELS = {
    # Production (51-xxxx)
    "51-1": "Production supervisors",
    "51-2": "Assembly and fabrication",
    "51-3": "Food processing",
    "51-4": "Metal and plastic work",
    "51-40": "Metal and plastic machine operators (extruding, forging, cutting, molding)",
    "51-41": "Welding, heat treating, and other metalwork",
    "51-5": "Printing",
    "51-6": "Textile, apparel, and furnishings",
    "51-7": "Woodworking",
    "51-8": "Plant and system operators",
    "51-9": "Other production (inspectors, testers, painting, etc.)",
    "51-90": "Chemical, mixing, cutting, inspecting, and lab technicians",
    "51-91": "Packaging, painting, CNC, semiconductor, and other production",
    # Construction (47-xxxx)
    "47-1": "Construction supervisors",
    "47-2": "Construction trades (carpenters, electricians, plumbers, etc.)",
    "47-201": "Masonry and stonework (boilermakers, masons)",
    "47-204": "Carpenters, floor layers, and tile setters",
    "47-205": "Concrete, terrazzo, and general construction labor",
    "47-207": "Site work (paving, equipment operators, drywall, taping)",
    "47-21": "Finishing trades (glaziers, insulation, painters, plumbers, roofers)",
    "47-22": "Metal and solar (sheet metal, structural iron, solar installers)",
    "47-3": "Construction helpers",
    "47-4": "Other construction (highway, rail, fencing, etc.)",
    "47-5": "Extraction (mining, drilling, blasting)",
    # Installation/Repair (49-xxxx)
    "49-1": "Supervisors of repair workers",
    "49-2": "Electrical and electronic equipment repair",
    "49-3": "Vehicle and mobile equipment mechanics",
    "49-9": "Other installation and repair",
    # Transportation (53-xxxx)
    "53-1": "Transportation supervisors",
    "53-2": "Air transportation",
    "53-3": "Motor vehicle operators",
    "53-4": "Rail transportation",
    "53-5": "Water transportation",
    "53-6": "Material moving (crane, conveyor, etc.)",
    "53-7": "Material moving (hand, laborers, etc.)",
    # Tech/Science/Engineering (15, 17, 19)
    "15-1": "Computer and information technology",
    "15-2": "Mathematical science",
    "17-1": "Architects and surveyors",
    "17-2": "Engineers",
    "17-20": "Engineers (aerospace, agricultural, biomedical, chemical, civil, computer, electrical, environmental)",
    "17-21": "Engineers (health/safety, industrial, marine, materials, mechanical, mining, nuclear, petroleum)",
    "17-3": "Engineering technicians and drafters",
    "19-1": "Life scientists",
    "19-2": "Physical scientists",
    "19-3": "Social scientists and psychologists",
    "19-4": "Science technicians",
    # Education (25-xxxx)
    "25-1": "Postsecondary teachers (college/university)",
    # Finer postsecondary teacher splits (5-digit)
    "25-10": "STEM professors (business, computer, math, engineering, science)",
    "25-11": "Humanities, arts, social science, and law professors",
    "25-12": "Engineering and architecture professors",
    "25-13": "Life and physical science professors",
    "25-14": "Life and physical science professors",
    "25-15": "Social science and law professors",
    "25-16": "Social science and law professors",
    "25-17": "Arts, humanities, and education professors",
    "25-18": "Arts, humanities, and education professors",
    "25-19": "Other postsecondary teachers and instructors",
    # Even finer postsecondary teacher splits (6-digit) for the 25-10xx block
    "25-101": "Business, computer science, math, architecture, and engineering professors",
    "25-104": "Agricultural, biological, and environmental science professors",
    "25-105": "Physical science professors (chemistry, physics, earth science)",
    "25-106": "Social science professors (economics, psychology, sociology, etc.)",
    "25-107": "Health and nursing professors",
    "25-108": "Education and library science professors",
    "25-2": "Preschool through secondary teachers",
    "25-3": "Other teachers and tutors",
    "25-4": "Librarians and media specialists",
    "25-9": "Other education occupations",
    # Healthcare clinical (29-xxxx)
    "29-1": "Physicians, dentists, and other practitioners",
    # Finer clinical splits (5-digit)
    "29-10": "Dentists, chiropractors, optometrists, and podiatrists",
    "29-11": "Therapists (physical, occupational, speech, respiratory, etc.)",
    "29-12": "Physicians and surgeons",
    "29-13": "Physician assistants and nurse practitioners",
    "29-2": "Health technologists and technicians",
    "29-9": "Other healthcare practitioners",
    # Office/Admin (43-xxxx)
    "43-1": "Office supervisors",
    "43-2": "Communications equipment operators",
    "43-3": "Financial clerks",
    "43-4": "Information and records clerks",
    "43-5": "Material recording, scheduling, and dispatching",
    "43-6": "Secretaries and administrative assistants",
    "43-9": "Other office and administrative support",
    # Creative/Media (27-xxxx)
    "27-1": "Art and design",
    "27-2": "Entertainers, performers, and athletes",
    "27-3": "Media and communication",
    "27-4": "Media and communication equipment workers",
    # Sales (41-xxxx)
    "41-1": "Sales supervisors",
    "41-2": "Retail sales",
    "41-3": "Sales representatives (services)",
    "41-4": "Sales representatives (wholesale/manufacturing)",
    "41-9": "Other sales",
    # Business/Financial (13-xxxx)
    "13-1": "Business operations specialists",
    "13-2": "Financial specialists",
    # Management (11-xxxx)
    "11-1": "Top executives",
    "11-2": "Advertising, marketing, PR, and sales managers",
    "11-3": "Operations specialty managers",
    "11-9": "Other management occupations",
    # Protective Service (33-xxxx)
    "33-1": "Protective service supervisors",
    "33-2": "Fire fighters and investigators",
    "33-3": "Law enforcement",
    "33-9": "Other protective service",
    # Personal Care (39-xxxx)
    "39-1": "Personal care supervisors",
    "39-2": "Animal care and service",
    "39-3": "Entertainment attendants",
    "39-4": "Funeral service",
    "39-5": "Personal appearance",
    "39-6": "Baggage porters and concierges",
    "39-7": "Tour guides and recreation",
    "39-9": "Other personal care",
    # Food Prep (35-xxxx)
    "35-1": "Food service supervisors",
    "35-2": "Cooks",
    "35-3": "Food and beverage servers",
    "35-9": "Other food preparation",
    # Farming (45-xxxx)
    "45-1": "Farming, fishing, forestry supervisors",
    "45-2": "Agricultural workers",
    "45-3": "Fishing and hunting",
    "45-4": "Forest, conservation, and logging",
    # Grounds (37-xxxx)
    "37-1": "Grounds maintenance supervisors",
    "37-2": "Grounds maintenance workers",
    "37-3": "Pest control and janitors",
    # Community/Social (21-xxxx)
    "21-1": "Counselors, social workers, and specialists",
    "21-2": "Religious workers",
}


def load_benchmarks(filepath):
    """Load occupation benchmarks (751 SOC codes + titles)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    occupations = {}
    for soc, v in data["occupations"].items():
        occupations[soc] = {
            "title": v["title"],
            "typical_education": v.get("typical_education", ""),
        }
    return occupations


def load_job_zones(filepath):
    """
    Load O*NET Job Zones and map to BLS 6-digit codes.

    Prefer .00 codes; average and round for specialties without .00.
    """
    onet_zones = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = row["O*NET-SOC Code"].strip()
            zone = int(row["Job Zone"])
            onet_zones[code] = zone

    # Aggregate to BLS 6-digit
    bls_groups = defaultdict(list)
    for code, zone in onet_zones.items():
        bls_code = code.split(".")[0]
        suffix = code.split(".")[1] if "." in code else "00"
        bls_groups[bls_code].append({"suffix": suffix, "zone": zone})

    bls_zones = {}
    for bls_code, entries in bls_groups.items():
        standard = [e for e in entries if e["suffix"] == "00"]
        if standard:
            bls_zones[bls_code] = standard[0]["zone"]
        else:
            avg = sum(e["zone"] for e in entries) / len(entries)
            bls_zones[bls_code] = round(avg)

    return bls_zones


def load_descriptions(filepath):
    """
    Load occupation descriptions from O*NET.

    Returns first sentence, truncated to ~150 chars.
    """
    onet_descs = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = row["O*NET-SOC Code"].strip()
            desc = row["Description"].strip()
            onet_descs[code] = desc

    # Aggregate to BLS 6-digit (prefer .00)
    bls_groups = defaultdict(list)
    for code, desc in onet_descs.items():
        bls_code = code.split(".")[0]
        suffix = code.split(".")[1] if "." in code else "00"
        bls_groups[bls_code].append({"suffix": suffix, "desc": desc})

    bls_descs = {}
    for bls_code, entries in bls_groups.items():
        standard = [e for e in entries if e["suffix"] == "00"]
        if standard:
            full_desc = standard[0]["desc"]
        else:
            full_desc = entries[0]["desc"]

        # Extract first sentence, truncate
        first_sent = full_desc.split(". ")[0]
        if len(first_sent) > 150:
            first_sent = first_sent[:147] + "..."
        elif not first_sent.endswith("."):
            first_sent += "."
        bls_descs[bls_code] = first_sent

    return bls_descs


def get_soc_4digit_label(soc_code):
    """Get a human-readable label for a 4-digit SOC group."""
    prefix_3 = soc_code[:4]  # e.g., "51-4" from "51-4121"
    if prefix_3 in SOC_4DIGIT_LABELS:
        return SOC_4DIGIT_LABELS[prefix_3]
    return f"SOC {prefix_3}xxx"


def classify_occupation(soc, zone):
    """
    Classify an occupation through Q1 → Q2 → Q3.

    Returns: (q1_id, q2_id, q3_id)
    """
    prefix_2 = soc[:2]

    # Q1: Work setting
    q1_id = None
    for branch_id, branch in Q1_BRANCHES.items():
        if prefix_2 in branch["soc_groups"]:
            q1_id = branch_id
            break

    if q1_id is None:
        print(f"  WARNING: SOC {soc} (prefix {prefix_2}) not mapped to any Q1 branch")
        return None

    # Q2: Domain
    q2_id = None
    for option_id, option in Q2_BRANCHES[q1_id]["options"].items():
        if prefix_2 in option["soc_groups"]:
            q2_id = option_id
            break

    if q2_id is None:
        print(f"  WARNING: SOC {soc} not mapped to any Q2 branch under {q1_id}")
        return None

    # Q3: Education/Job Zone
    q3_id = None
    for option_id, option in Q3_OPTIONS.items():
        if zone in option["zones"]:
            q3_id = option_id
            break

    if q3_id is None:
        print(f"  WARNING: SOC {soc} zone {zone} not mapped to any Q3 option")
        return None

    return (q1_id, q2_id, q3_id)


def split_soc_group(soc_list, prefix_len=4, min_group_size=3):
    """
    Split a list of SOC codes by their prefix at the given length.

    Args:
        soc_list: list of SOC codes
        prefix_len: how many characters of SOC code to group by (4 or 5)
        min_group_size: minimum group size before merging (default 3)

    Returns:
        list of (prefix, soc_list) tuples, with small groups merged
    """
    subgroups = defaultdict(list)
    for soc in soc_list:
        prefix = soc[:prefix_len]
        subgroups[prefix].append(soc)

    sorted_groups = sorted(subgroups.items())

    # Merge small groups with adjacent small groups first, then with
    # nearest larger group. This avoids collapsing many small groups
    # into one big group.
    merged = []
    pending_prefix = None
    pending_socs = []

    for prefix, socs in sorted_groups:
        if len(socs) < min_group_size:
            # Accumulate small groups
            if pending_prefix is None:
                pending_prefix = prefix
            pending_socs.extend(socs)
            # If accumulated enough, flush as a group
            if len(pending_socs) >= min_group_size:
                merged.append((pending_prefix, pending_socs))
                pending_prefix = None
                pending_socs = []
        else:
            # Large enough group — flush any pending first
            if pending_socs:
                if len(pending_socs) >= min_group_size:
                    # Pending is big enough on its own
                    merged.append((pending_prefix, pending_socs))
                else:
                    # Attach pending to this group
                    socs = pending_socs + socs
                pending_prefix = None
                pending_socs = []
            merged.append((prefix, socs))

    # Handle any remaining pending
    if pending_socs:
        if merged:
            if len(pending_socs) >= min_group_size:
                merged.append((pending_prefix, pending_socs))
            else:
                last_prefix, last_socs = merged[-1]
                merged[-1] = (last_prefix, last_socs + pending_socs)
        else:
            merged.append((pending_prefix or sorted_groups[0][0], pending_socs))

    return merged


def get_label_for_prefix(prefix, soc_list):
    """Get best human-readable label for a SOC prefix group."""
    # Try 6-char prefix first (finest grain, e.g., "25-101")
    if len(prefix) >= 6:
        prefix_6 = prefix[:6]
        if prefix_6 in SOC_4DIGIT_LABELS:
            return SOC_4DIGIT_LABELS[prefix_6]

    # Try 5-char prefix (for finer splits like "25-10")
    if len(prefix) >= 5:
        prefix_5 = prefix[:5]
        if prefix_5 in SOC_4DIGIT_LABELS:
            return SOC_4DIGIT_LABELS[prefix_5]

    # Try 4-char prefix (standard, e.g., "51-4")
    normalized = prefix[:4]
    if normalized in SOC_4DIGIT_LABELS:
        return SOC_4DIGIT_LABELS[normalized]

    # Fallback: use first SOC code's 4-digit label
    return SOC_4DIGIT_LABELS.get(soc_list[0][:4], f"SOC {prefix}xxx")


def build_q4_splits(oversized_leaves, occupations, descriptions):
    """
    Build Q4 specialization splits for oversized leaf nodes.

    Strategy:
    1. First split by 4-digit SOC prefix (e.g., "51-4" from "51-4121")
    2. If any resulting group is still > MAX_LEAF_SIZE, split further
       by 5-digit prefix (e.g., "25-10" from "25-1011")
    3. Merge groups that are still > MAX_LEAF_SIZE with adjacent groups
       as a last resort (some large homogeneous groups like "construction
       trades" naturally have 20+ occupations — this is acceptable)
    """
    q4_branches = {}

    for leaf_key, soc_list in oversized_leaves.items():
        # First attempt: split by 4-digit prefix
        groups_4 = split_soc_group(soc_list, prefix_len=4)

        # Check if any group is still too large — try progressively finer splits
        final_groups = []
        for prefix, socs in groups_4:
            if len(socs) > MAX_LEAF_SIZE:
                # Try finer 5-digit splitting
                sub_groups = split_soc_group(socs, prefix_len=5)
                if len(sub_groups) > 1:
                    # 5-digit split worked — check if any sub-group still too large
                    for sub_prefix, sub_socs in sub_groups:
                        if len(sub_socs) > MAX_LEAF_SIZE:
                            # Try even finer 6-digit splitting
                            sub_sub = split_soc_group(sub_socs, prefix_len=6)
                            if len(sub_sub) > 1:
                                for ss_prefix, ss_socs in sub_sub:
                                    final_groups.append((ss_prefix, ss_socs))
                            else:
                                final_groups.append((sub_prefix, sub_socs))
                        else:
                            final_groups.append((sub_prefix, sub_socs))
                else:
                    # Can't split further — keep as-is
                    final_groups.append((prefix, socs))
            else:
                final_groups.append((prefix, socs))

        # Second pass: merge any remaining tiny groups
        merged_final = []
        pending = []
        for prefix, socs in final_groups:
            if len(socs) < 3:
                pending.extend(socs)
            else:
                if pending:
                    socs = pending + socs
                    pending = []
                merged_final.append((prefix, socs))
        if pending:
            if merged_final:
                last_prefix, last_socs = merged_final[-1]
                merged_final[-1] = (last_prefix, last_socs + pending)
            else:
                merged_final.append((final_groups[0][0], pending))

        # If we ended up with only 1 group, try 5-digit across the whole leaf
        if len(merged_final) <= 1 and len(soc_list) > MAX_LEAF_SIZE:
            groups_5 = split_soc_group(soc_list, prefix_len=5)
            if len(groups_5) > 1:
                merged_final = groups_5

        # Build Q4 options
        options = []
        for prefix, socs in merged_final:
            option_id = prefix.replace("-", "_").lower()
            label = get_label_for_prefix(prefix, socs)
            options.append({
                "id": option_id,
                "label": label,
                "soc_codes": sorted(socs),
            })

        q4_text = "Which area most closely matches your specialty?"

        q4_branches[leaf_key] = {
            "text": q4_text,
            "options": options,
        }

    return q4_branches


def run_sanity_checks(leaves, all_socs, occupations, search_index):
    """Run comprehensive sanity checks."""
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

    # 1. All occupations assigned
    assigned = set()
    for leaf_data in leaves.values():
        for occ in leaf_data["occupations"]:
            assigned.add(occ["soc"])
    check(
        "All 751 occupations assigned to exactly one leaf",
        len(assigned) == len(all_socs) and assigned == set(all_socs),
        f"{len(assigned)} assigned, {len(all_socs)} expected",
    )

    # 2. No empty leaves
    empty = [k for k, v in leaves.items() if v["count"] == 0]
    check("No empty leaves", len(empty) == 0, f"{len(empty)} empty" if empty else "")

    # 3. Leaf size distribution
    sizes = [v["count"] for v in leaves.values()]
    import numpy as np
    median_size = np.median(sizes)
    check(
        "Median leaf size in 5-15 range",
        5 <= median_size <= 15,
        f"median: {median_size:.1f}",
    )

    # 4. Max leaf size
    max_size = max(sizes)
    max_leaf = max(leaves.items(), key=lambda x: x[1]["count"])
    check(
        f"Max leaf size ≤ {MAX_LEAF_SIZE + 5}",
        max_size <= MAX_LEAF_SIZE + 5,
        f"max: {max_size} ({max_leaf[0]})",
    )

    # 5. Percentage in 3-15 range
    in_range = sum(1 for s in sizes if 3 <= s <= 15)
    pct = in_range / len(sizes) * 100
    check(
        "≥ 60% of leaves in 3-15 range",
        pct >= 60,
        f"{pct:.1f}% ({in_range}/{len(sizes)})",
    )

    # 6. Max depth ≤ 4
    max_depth = max(len(v["path"]) for v in leaves.values())
    check("All paths reachable in ≤ 4 questions", max_depth <= 4, f"max depth: {max_depth}")

    # 7. No duplicates
    all_assigned = []
    for leaf_data in leaves.values():
        all_assigned.extend(occ["soc"] for occ in leaf_data["occupations"])
    check(
        "No duplicate SOC codes across leaves",
        len(all_assigned) == len(set(all_assigned)),
        f"{len(all_assigned)} total, {len(set(all_assigned))} unique",
    )

    # 8. Spot check: Software Developers
    if "15-1252" in search_index:
        sw_leaf = search_index["15-1252"]["leaf"]
        sw_path = leaves[sw_leaf]["path"]
        expected = sw_path[0] == "office" and sw_path[1] == "tech_science"
        check(
            "Software Developers → office.tech_science.*",
            expected,
            f"path: {'.'.join(sw_path)}",
        )
    else:
        print("  [SKIP] Software Developers not in search index")

    # 9. Spot check: Registered Nurses
    if "29-1141" in search_index:
        rn_leaf = search_index["29-1141"]["leaf"]
        rn_path = leaves[rn_leaf]["path"]
        expected = rn_path[0] == "healthcare" and rn_path[1] == "clinical"
        check(
            "Registered Nurses → healthcare.clinical.*",
            expected,
            f"path: {'.'.join(rn_path)}",
        )
    else:
        print("  [SKIP] Registered Nurses not in search index")

    # 10. Spot check: Carpenters
    if "47-2031" in search_index:
        carp_leaf = search_index["47-2031"]["leaf"]
        carp_path = leaves[carp_leaf]["path"]
        expected = carp_path[0] == "trades" and carp_path[1] == "construction"
        check(
            "Carpenters → trades.construction.*",
            expected,
            f"path: {'.'.join(carp_path)}",
        )
    else:
        print("  [SKIP] Carpenters not in search index")

    # 11. Distribution summary
    print(f"\n  Leaf size distribution ({len(sizes)} leaves):")
    print(f"    Min: {min(sizes)}  Max: {max(sizes)}  Median: {median_size:.1f}  Mean: {np.mean(sizes):.1f}")
    brackets = [(1, 2), (3, 5), (6, 10), (11, 15), (16, 20), (21, 999)]
    for lo, hi in brackets:
        count = sum(1 for s in sizes if lo <= s <= hi)
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        bar = "█" * count
        print(f"    {label:>5}: {count:3d} {bar}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Verify inputs
    for f, label in [(BENCHMARKS_FILE, "Benchmarks"), (JOB_ZONES_FILE, "Job Zones"),
                     (OCC_DATA_FILE, "Occupation Data")]:
        if not f.exists():
            print(f"ERROR: {label} not found: {f}")
            sys.exit(1)

    print("Step 8: Build Quiz-Filter Decision Tree")
    print("=" * 60)

    # ================================================================
    # 1. Load data
    # ================================================================
    print("\n1. Loading data...")
    occupations = load_benchmarks(BENCHMARKS_FILE)
    print(f"  Loaded {len(occupations)} occupations from benchmarks")

    job_zones = load_job_zones(JOB_ZONES_FILE)
    zone_coverage = sum(1 for soc in occupations if soc in job_zones)
    print(f"  Job Zones: {zone_coverage}/{len(occupations)} coverage")

    descriptions = load_descriptions(OCC_DATA_FILE)
    desc_coverage = sum(1 for soc in occupations if soc in descriptions)
    print(f"  Descriptions: {desc_coverage}/{len(occupations)} coverage")

    # Handle missing zones — assign based on typical_education as fallback
    edu_to_zone = {
        "No formal educational credential": 1,
        "High school diploma or equivalent": 2,
        "Some college, no degree": 2,
        "Postsecondary nondegree award": 3,
        "Associate's degree": 3,
        "Bachelor's degree": 4,
        "Master's degree": 5,
        "Doctoral or professional degree": 5,
    }
    missing_zones = []
    for soc in occupations:
        if soc not in job_zones:
            edu = occupations[soc]["typical_education"]
            fallback_zone = edu_to_zone.get(edu, 2)
            job_zones[soc] = fallback_zone
            missing_zones.append((soc, edu, fallback_zone))

    if missing_zones:
        print(f"  Assigned fallback zones for {len(missing_zones)} occupations:")
        for soc, edu, zone in missing_zones[:5]:
            print(f"    {soc}: {edu} → Zone {zone}")

    # ================================================================
    # 2. Classify all occupations through Q1→Q2→Q3
    # ================================================================
    print("\n2. Classifying occupations through Q1→Q2→Q3...")

    # Build leaf buckets: key = "q1.q2.q3" → list of SOC codes
    leaf_buckets = defaultdict(list)
    unclassified = []

    for soc in sorted(occupations.keys()):
        zone = job_zones.get(soc)
        if zone is None:
            unclassified.append(soc)
            continue

        result = classify_occupation(soc, zone)
        if result is None:
            unclassified.append(soc)
            continue

        q1_id, q2_id, q3_id = result
        leaf_key = f"{q1_id}.{q2_id}.{q3_id}"
        leaf_buckets[leaf_key].append(soc)

    if unclassified:
        print(f"  WARNING: {len(unclassified)} unclassified occupations:")
        for soc in unclassified:
            print(f"    {soc}: {occupations[soc]['title']}")

    total_assigned = sum(len(v) for v in leaf_buckets.values())
    print(f"  Classified {total_assigned} occupations into {len(leaf_buckets)} leaf nodes")

    # ================================================================
    # 3. Identify oversized leaves and build Q4 splits
    # ================================================================
    print("\n3. Checking for oversized leaves (>{MAX_LEAF_SIZE})...")

    oversized = {}
    normal = {}
    for key, soc_list in leaf_buckets.items():
        if len(soc_list) > MAX_LEAF_SIZE:
            oversized[key] = soc_list
        else:
            normal[key] = soc_list

    print(f"  Normal leaves (≤{MAX_LEAF_SIZE}): {len(normal)}")
    print(f"  Oversized leaves (>{MAX_LEAF_SIZE}): {len(oversized)}")

    if oversized:
        print(f"\n  Oversized leaves:")
        for key in sorted(oversized, key=lambda k: -len(oversized[k])):
            print(f"    {key}: {len(oversized[key])} occupations")

    # Build Q4 splits
    q4_branches = build_q4_splits(oversized, occupations, descriptions)

    # ================================================================
    # 4. Build final leaves dict
    # ================================================================
    print("\n4. Building final leaf structure...")

    leaves = {}

    # Add normal (non-Q4) leaves
    for key, soc_list in normal.items():
        path = key.split(".")
        occ_list = []
        for soc in sorted(soc_list):
            occ_list.append({
                "soc": soc,
                "title": occupations[soc]["title"],
                "description": descriptions.get(soc, ""),
            })
        leaves[key] = {
            "occupations": occ_list,
            "count": len(occ_list),
            "path": path,
        }

    # Add Q4-split leaves
    for parent_key, branch in q4_branches.items():
        for option in branch["options"]:
            q4_id = option["id"]
            leaf_key = f"{parent_key}.{q4_id}"
            path = parent_key.split(".") + [q4_id]
            occ_list = []
            for soc in sorted(option["soc_codes"]):
                occ_list.append({
                    "soc": soc,
                    "title": occupations[soc]["title"],
                    "description": descriptions.get(soc, ""),
                })
            leaves[leaf_key] = {
                "occupations": occ_list,
                "count": len(occ_list),
                "path": path,
            }

    # Remove any empty leaves that might have been created
    leaves = {k: v for k, v in leaves.items() if v["count"] > 0}

    print(f"  Total leaves: {len(leaves)}")

    # ================================================================
    # 5. Build search index
    # ================================================================
    print("\n5. Building search index...")

    search_index = {}
    for leaf_key, leaf_data in leaves.items():
        for occ in leaf_data["occupations"]:
            search_index[occ["soc"]] = {
                "title": occ["title"],
                "leaf": leaf_key,
            }

    print(f"  Indexed {len(search_index)} occupations")

    # ================================================================
    # 6. Sanity checks
    # ================================================================
    all_socs = sorted(occupations.keys())
    all_pass = run_sanity_checks(leaves, all_socs, occupations, search_index)

    if not all_pass:
        print("\nWARNING: Some sanity checks failed. Review output carefully.")

    # ================================================================
    # 7. Build output JSON
    # ================================================================
    print("\n" + "=" * 60)
    print("WRITING OUTPUT")
    print("=" * 60)

    sizes = [v["count"] for v in leaves.values()]

    # Build questions structure
    questions = {
        "q1": {
            "text": "Which best describes where and how you work?",
            "options": [
                {"id": bid, "label": b["label"]}
                for bid, b in Q1_BRANCHES.items()
            ],
        },
        "q2": {
            "depends_on": "q1",
            "branches": {},
        },
        "q3": {
            "text": "What level of education is typical for your role?",
            "options": [
                {"id": oid, "label": o["label"]}
                for oid, o in Q3_OPTIONS.items()
            ],
        },
    }

    # Build Q2 branches
    for q1_id, q2_def in Q2_BRANCHES.items():
        questions["q2"]["branches"][q1_id] = {
            "text": q2_def["text"],
            "options": [
                {"id": oid, "label": o["label"]}
                for oid, o in q2_def["options"].items()
            ],
        }

    # Build Q4 branches (if any)
    if q4_branches:
        questions["q4"] = {
            "depends_on": ["q1", "q2", "q3"],
            "branches": {},
        }
        for parent_key, branch in q4_branches.items():
            questions["q4"]["branches"][parent_key] = {
                "text": branch["text"],
                "options": [
                    {"id": opt["id"], "label": opt["label"]}
                    for opt in branch["options"]
                ],
            }

    # Build valid_q3_options: for each Q1+Q2 combo, list which Q3 options
    # actually have occupations. This lets the front-end dynamically filter
    # Q3 choices rather than showing options that lead to empty results.
    valid_q3 = {}
    for q1_id in Q1_BRANCHES:
        for q2_id in Q2_BRANCHES[q1_id]["options"]:
            key = f"{q1_id}.{q2_id}"
            valid = []
            for q3_id in Q3_OPTIONS:
                path_3 = f"{q1_id}.{q2_id}.{q3_id}"
                has_occs = path_3 in leaves or path_3 in q4_branches
                if has_occs:
                    valid.append(q3_id)
            valid_q3[key] = valid

    total_combos = sum(len(Q2_BRANCHES[q1]["options"]) for q1 in Q1_BRANCHES)
    empty_combos = sum(1 for v in valid_q3.values() if len(v) < len(Q3_OPTIONS))
    print(f"\n  Valid Q3 map: {len(valid_q3)} Q1+Q2 combos, {empty_combos} have restricted Q3 options")

    import numpy as np
    output = {
        "metadata": {
            "description": "Quiz-filter decision tree for occupation selection",
            "total_occupations": len(all_socs),
            "total_leaves": len(leaves),
            "max_depth": max(len(v["path"]) for v in leaves.values()),
            "leaf_size_stats": {
                "min": int(min(sizes)),
                "max": int(max(sizes)),
                "median": round(float(np.median(sizes)), 1),
                "mean": round(float(np.mean(sizes)), 1),
            },
            "q4_branches_count": len(q4_branches),
            "frontend_notes": {
                "q3_dynamic_filtering": (
                    "Use valid_q3_options to show only education levels that "
                    "have matching occupations for the selected Q1+Q2 path."
                ),
                "q4_option_count": (
                    "4 of 18 Q4 branches have 8-9 options. Consider a "
                    "scrollable list or two-column layout for these. "
                    "Revisit during design/UAT phase."
                ),
                "small_leaves": (
                    "8 leaves have 1-2 occupations (niche roles). Render "
                    "gracefully — the user simply sees a short list."
                ),
                "empty_path_handling": (
                    "18 Q1+Q2+Q3 combos have no occupations. The "
                    "valid_q3_options map prevents users from reaching "
                    "these by filtering Q3 choices dynamically."
                ),
            },
            "generated_by": "scripts/08_build_filter_tree.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "questions": questions,
        "valid_q3_options": valid_q3,
        "leaves": leaves,
        "search_index": search_index,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n  Wrote {OUTPUT_FILE} ({file_size:.2f} MB)")

    # ================================================================
    # 8. Sample output
    # ================================================================
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    # Show 5 example paths
    sample_leaves = sorted(leaves.items(), key=lambda x: -x[1]["count"])
    print("\n  5 largest leaves:")
    for key, leaf in sample_leaves[:5]:
        print(f"    {key} ({leaf['count']} occupations):")
        for occ in leaf["occupations"][:3]:
            print(f"      {occ['soc']} — {occ['title']}")
        if leaf["count"] > 3:
            print(f"      ... and {leaf['count'] - 3} more")

    # Show 5 smallest leaves
    print("\n  5 smallest leaves:")
    for key, leaf in sample_leaves[-5:]:
        print(f"    {key} ({leaf['count']} occupations):")
        for occ in leaf["occupations"]:
            print(f"      {occ['soc']} — {occ['title']}")

    # Show spotlight occupation paths
    spotlight = [
        ("15-1252", "Software Developers"),
        ("29-1141", "Registered Nurses"),
        ("47-2031", "Carpenters"),
        ("11-1021", "General Managers"),
        ("43-4051", "Customer Service Reps"),
    ]
    print("\n  Spotlight occupation paths:")
    for soc, name in spotlight:
        if soc in search_index:
            leaf_key = search_index[soc]["leaf"]
            path = leaves[leaf_key]["path"]
            leaf_size = leaves[leaf_key]["count"]
            print(f"    {soc} ({name}): {' → '.join(path)} ({leaf_size} in leaf)")
        else:
            print(f"    {soc} ({name}): NOT FOUND")

    print(f"\n  File size: {file_size:.2f} MB")


if __name__ == "__main__":
    main()
