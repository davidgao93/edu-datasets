# CLAUDE.md — BI Dataset Generator

## Project Overview

This is a **custom BI Dataset Generator** built for an e-learning platform (Articulate Rise + MicroStrategy). It produces personalized, industry-skinned datasets for learners across Retail, FSI, Life Sciences, Public Sector, and SaaS verticals — all sharing identical underlying math so a single answer key works across all industries.

## Core Architecture: Isomorphic Data Architecture

The central design principle is a strict separation of math from vocabulary:

- **The Skeleton (`app.py`):** Generates a Star Schema with mathematically locked integer IDs, sorted dates, and "Smart Math" (Pareto-distributed item weights, inverse price/volume correlation via `base_p * base_v`, Q4 seasonality multiplier of 1.5). `np.random.seed(42)` and `Faker.seed(42)` are both locked — **never change these seeds.**
- **The Skin (`config.json`):** Maps the skeleton's generic field keys (e.g., `ent_lvl3`, `vol`, `fin`) to industry-specific column headers (e.g., `Store` / `Clinic` / `Branch`). Also holds dimension member lists (entity groups, item groups, tiers, roles, channels).
- **The Guarantee:** Because the seed is locked and the math is identical, a student in Healthcare and a student in Retail will compute the exact same numerical answer. This is what makes a single answer key possible.

## File Structure

```
edu-datasets/
├── app.py          # Streamlit app — UI, data generation engine, export
├── config.json     # Industry skins (headers + dimension members)
└── CLAUDE.md       # This file
```

## Tech Stack

| Tool | Role |
|---|---|
| Python / Streamlit | Web UI (`app.py`) |
| Pandas / NumPy | Star Schema generation and Smart Math |
| Faker (seed=42) | Realistic names for Customer_Name, Employee_Name, and city-based entity names |
| JSON (`config.json`) | Decoupled industry configuration |
| Streamlit Data Editor | In-browser metadata customization; PKs are locked |
| `io` / `zipfile` | In-memory ZIP export of all CSVs |

## Star Schema Tables

| Table | Rows | PK | Notes |
|---|---|---|---|
| `Fact_Transactions` | 5k–50k (slider) | `Transaction_ID` (100000+) | Sorted by date; read-only in UI |
| `Dim_Entity` | ~40 | `Entity_ID` (101+) | 4 groups × 10 Faker cities |
| `Dim_Item` | ~40 | `Item_ID` (2001+) | 4 groups × 10 tier modifiers; drives Pareto price/volume |
| `Dim_Customer` | 1,000 | `cust_id` (10001+) | Faker names; age brackets; industry-specific tiers |
| `Dim_Employee` | 50 | `emp_id` (501+) | Faker names; role drives `_perf_weight` (hidden from export) |
| `Dim_Channel` | 3 | `Channel_ID` (1+) | |
| `Dim_Date` | 730 | `Date` | 2023-01-01 to 2024-12-31; read-only in UI |

## Smart Math Rules (DO NOT ALTER)

These relationships must remain intact for answer keys to stay valid:

1. **Pareto item weights** — `np.random.pareto(a=2.0)` drives item selection probability, creating realistic long-tail distributions.
2. **Inverse price/volume** — Higher-priced items have lower base volume because `base_v` (Pareto draw) is independent of `base_p` (uniform $50–$2000). This is intentional.
3. **Employee performance** — `_perf_weight` maps roles to multipliers: Trainee/Junior/Clerk = 0.5, mid = 1.0, senior/manager/VP = 1.5.
4. **Q4 Seasonality** — `season_mult = 1.5` for months >= 11 (November, December).
5. **Noise** — Volume noise: `uniform(0.8, 1.2)`. Price noise: `uniform(0.95, 1.05)`.
6. **Final metrics** — `vol = max(1, base_v * emp_w * season_mult * noise_v * 10).astype(int)` and `fin = vol * (base_p * noise_p)`.

## config.json Schema

Each industry key contains:

```json
{
  "headers": {
    "ent_lvl1": "...", "ent_lvl2": "...", "ent_lvl3": "...",
    "item_lvl1": "...", "item_lvl2": "...", "item_lvl3": "...",
    "cust_id": "...", "cust_tier": "...",
    "emp_id": "...", "emp_role": "...",
    "channel": "...",
    "vol": "...", "fin": "..."
  },
  "entity_groups": [["lvl1_val", "lvl2_val"], ...],
  "item_groups": [["lvl1_val", "lvl2_val"], ...],
  "customer_tiers": ["tier1", "tier2", "tier3"],
  "employee_roles": ["junior", "mid", "senior"],
  "channels": ["channel1", "channel2", "channel3"]
}
```

`entity_groups` and `item_groups` must each have exactly 4 pairs for the 4×10 dimension generation to produce ~40 rows. `employee_roles` must have exactly 3 entries (mapped to perf weights 0.5 / 1.0 / 1.5 in that order). `customer_tiers` and `channels` can vary in length.

## Current Industries

- **Retail** — Store / Units_Sold / Revenue
- **FSI** — Branch / New_Accounts / Assets_Under_Management
- **Life Sciences** — Clinic / Procedures_Done / Billed_Amount
- **Public Sector** — Office / Cases_Resolved / Funds_Disbursed
- **SaaS** — Hub / Seats_Sold / Annual_Recurring_Revenue

## Key Constraints

- **Never change `np.random.seed(42)` or `Faker.seed(42)`** — doing so invalidates all existing answer keys.
- **Primary Keys are intentionally locked** in `st.data_editor` to protect referential integrity between Fact and Dimension tables.
- **`_perf_weight`** is a helper column stripped before export — it must not appear in any exported CSV.
- The `generate_isomorphic_data` function is decorated with `@st.cache_data` — it only re-runs when `ind_name`, `rows`, or `cfg` changes.
- The Fact table is read-only in the UI; only Dimension tables are editable.

## Running the App

```bash
streamlit run app.py
```
