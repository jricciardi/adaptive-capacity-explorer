# Adaptive Capacity Self-Assessment Tool

**[Try it live](https://jricciardi.github.io/adaptive-capacity-explorer/)**

A browser-based tool that helps workers understand how prepared they are for AI-driven labor market transitions, based on peer-reviewed economics research.

## What this is

This is a research translation project. Economists at Brookings measured "adaptive capacity" across 752 occupations — a composite of skill transferability, geographic labor market density, financial reserves, and age — and published the framework in [Manning & Aguirre (2026)](https://www.nber.org/papers/w34705). This tool makes that framework explorable for individual workers.

You select your occupation, enter a few personal inputs, and get back a composite score with a breakdown of what's working for you and what isn't. All computation runs in your browser. Nothing is stored, sent, or tracked.

## Running locally

No build step. No dependencies. Just a static site:

```
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## How it works

The composite score combines four equally weighted components, converted to Z-scores and mapped to a percentile via normal CDF approximation:

| Component | Source | What it measures |
|---|---|---|
| **Skill transferability** | O\*NET skill profiles + BLS employment projections | How easily your occupation's skills map to other growing occupations |
| **Geographic density** | BLS OEWS + Census CBSA data | How many alternative employers exist in your labor market |
| **Financial runway** | User input (months of expenses covered) | How long you could search for a good-fit role rather than the first available |
| **Age factor** | User input | Time horizon for recouping retraining investments |

Transferability and density are occupation-level benchmarks from the study. Financial runway and age are personal inputs mapped to population distributions using simplified assumptions.

## Data pipeline

The `data/` directory contains pre-built JSON files, so running the pipeline is optional. If you want to rebuild from source:

**Requirements:** Python 3, numpy, openpyxl (`pip install numpy openpyxl`)

O\*NET 30.1 database files must be downloaded separately from [onetcenter.org](https://www.onetcenter.org/database.html) and extracted to `/tmp/onet_30_1/db_30_1_text/`.

Run scripts in order:

| Script | What it does |
|---|---|
| `01_parse_onet_skills.py` | Parses O\*NET 30.1 skill + work activity importance ratings into 76-dimension vectors per occupation |
| `02_compute_skill_similarity.py` | Percentile-rank normalizes profiles, computes pairwise cosine similarity across all occupations |
| `03_parse_bls_projections.py` | Downloads BLS 2024–2034 employment projections (employment, growth, wages, education) |
| `04_compute_transferability.py` | Implements the Manning & Aguirre transferability formula: employment-weighted similarity to growing occupations |
| `05_compute_density.py` | Implements the Manning & Aguirre density formula: employment-weighted average log(employment/area) across metros |
| `06_bundle_ai_exposure.py` | Downloads Eloundou et al. AI exposure scores from the GPTs-are-GPTs repository |
| `07_build_benchmarks.py` | Merges all sources into a single per-occupation benchmark table with composite scores |
| `08_build_filter_tree.py` | Builds a guided quiz decision tree for occupation selection |

## Data sources

| Source | Version |
|---|---|
| O\*NET | 30.1 (December 2025) |
| BLS Employment Projections | 2024–2034 |
| BLS OEWS | May 2024 |
| Census CBSA Gazetteer | 2024 |
| AI exposure scores | [Eloundou et al. (2024)](https://doi.org/10.1126/science.adj0998) |

## Limitations

- Skill transferability is measured by O\*NET skill profile similarity, which may not capture all relevant factors in career transitions.
- The composite score combines occupation-level benchmarks with self-reported personal inputs using simplified distributional assumptions for wealth and age.
- Occupation-level averages mask within-occupation variation. Two software developers may have very different adaptive capacity depending on their specific skills, employer, and network.
- Geographic density uses the occupation's national employment distribution, not the user's specific metro. The metro selection provides context but does not change the density score.

## Research

Manning, A. & Aguirre, J. (2026). "Who Bears the Costs of AI Displacement?" NBER Working Paper 34705. [nber.org/papers/w34705](https://www.nber.org/papers/w34705)

Eloundou, T., Manning, S., Mishkin, P. & Rock, D. (2024). "GPTs are GPTs: Labor Market Impact Potential of LLMs." *Science* 384(6702). [doi.org/10.1126/science.adj0998](https://doi.org/10.1126/science.adj0998)

## License

[MIT](LICENSE)
