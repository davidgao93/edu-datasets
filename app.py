import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker

st.set_page_config(page_title="EDU Dataset Generator", layout="wide")
st.title("📊 EDU Dataset Generator")


# --- 1. LOAD EXTERNAL CONFIGURATION ---
@st.cache_data
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)


industries = load_config()

# Industries present at first launch — protected from rename/delete in the management panel.
_BUILTIN_INDUSTRIES = {"Retail", "FSI", "Life Sciences", "Public Sector", "SaaS"}

_REQUIRED_HEADER_KEYS = [
    "ent_lvl1",
    "ent_lvl2",
    "ent_lvl3",
    "item_lvl1",
    "item_lvl2",
    "item_lvl3",
    "cust_id",
    "cust_tier",
    "emp_id",
    "emp_role",
    "channel",
    "vol",
    "fin",
]


def _validate_industry_config(data):
    errors = []
    for key in [
        "headers",
        "entity_groups",
        "item_groups",
        "customer_tiers",
        "employee_roles",
        "channels",
    ]:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")
    if errors:
        return errors

    for h in _REQUIRED_HEADER_KEYS:
        if h not in data["headers"]:
            errors.append(f"Missing header key: '{h}'")
        elif not isinstance(data["headers"][h], str) or not data["headers"][h].strip():
            errors.append(f"Header '{h}' must be a non-empty string")

    for field in ("entity_groups", "item_groups"):
        grps = data[field]
        if not isinstance(grps, list) or len(grps) != 4:
            errors.append(
                f"'{field}' must be a list of exactly 4 pairs "
                f"(got {len(grps) if isinstance(grps, list) else type(grps).__name__})"
            )
        else:
            for i, pair in enumerate(grps):
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not all(isinstance(s, str) for s in pair)
                ):
                    errors.append(f"'{field}[{i}]' must be a list of exactly 2 strings")

    roles = data["employee_roles"]
    if not isinstance(roles, list) or len(roles) != 3:
        errors.append(
            f"'employee_roles' must be a list of exactly 3 roles "
            f"(got {len(roles) if isinstance(roles, list) else type(roles).__name__})"
        )

    if not isinstance(data["customer_tiers"], list) or len(data["customer_tiers"]) < 2:
        errors.append("'customer_tiers' must be a list of at least 2 tiers")

    if not isinstance(data["channels"], list) or len(data["channels"]) < 1:
        errors.append("'channels' must be a list of at least 1 channel")

    return errors


# --- 2. TOP BAR CONTROLS ---
_ADD_SENTINEL = "➕ Add your own..."

col1, col2 = st.columns([1, 2])
with col1:
    selected_ind = st.selectbox(
        "1. Select Industry Context:",
        options=[_ADD_SENTINEL] + list(industries.keys()),
    )
    if selected_ind != _ADD_SENTINEL:
        num_rows = st.slider(
            "2. Number of Fact Rows:",
            min_value=5000,
            max_value=50000,
            value=10000,
            step=5000,
        )
        seed = st.number_input(
            "3. Random Seed",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            help=(
                "Change the seed to produce a completely different dataset. "
                "The Answer Key regenerates automatically to match, so any seed is valid."
            ),
        )
with col2:
    if selected_ind != _ADD_SENTINEL:
        dirty_mode = st.checkbox(
            "Enable Dirty Data Mode",
            help=(
                "Injects mathematically locked nulls, duplicates, and outliers "
                "into the export for data cleansing / wrangling exercises."
            ),
        )
        if dirty_mode:
            st.caption("Configure injection levels:")
            null_pct = st.slider(
                "Null Rate (%)", min_value=0, max_value=10, value=3,
                help="Percentage of Customer_Name values set to null in Dim_Customer.",
            )
            dup_count = st.slider(
                "Duplicate Rows", min_value=0, max_value=200, value=50, step=10,
                help="Number of exact duplicate rows injected into Fact_Transactions.",
            )
            outlier_mult = st.slider(
                "Outlier Multiplier", min_value=10, max_value=100, value=50, step=10,
                help="Vol and Fin for 20 outlier rows are multiplied by this factor.",
            )
        else:
            null_pct, dup_count, outlier_mult = 3, 50, 50


# --- 3. ADD YOUR OWN INDUSTRY ---
if selected_ind == _ADD_SENTINEL:
    st.divider()
    if "_success_msg" in st.session_state:
        st.success(st.session_state.pop("_success_msg"))
    st.subheader("Add New Industry")
    st.markdown(
        "Describe the industry you want to add. Copy the generated prompt into Claude, "
        "ChatGPT, or any LLM, then paste the JSON output back below to validate and save it."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        new_ind_name = st.text_input("Industry Name", placeholder="e.g., Logistics")
    with col_b:
        new_ind_desc = st.text_input(
            "Brief Description",
            placeholder="e.g., freight companies tracking shipments across warehouses",
        )

    if new_ind_name.strip() and new_ind_desc.strip():
        prompt_text = f"""You are generating a JSON configuration for a BI dataset generator used for education and training.

Industry: {new_ind_name}
Description: {new_ind_desc}

Return ONLY a valid JSON object — no markdown fences, no explanation, no extra text. Use this exact structure:

{{
  "headers": {{
    "ent_lvl1": "Top-level geographic or organizational grouping (e.g. Continent, Territory, Network)",
    "ent_lvl2": "Second-level grouping (e.g. Region, District, Division)",
    "ent_lvl3": "Individual location name suffix — one word (e.g. Store, Branch, Clinic, Warehouse)",
    "item_lvl1": "Top-level product/service category (e.g. Department, Division, Suite)",
    "item_lvl2": "Second-level product/service category (e.g. Category, Product_Line, Module)",
    "item_lvl3": "Individual product/service name (e.g. Product, Service, SKU, Procedure)",
    "cust_id": "Customer/client ID column name (e.g. Customer_ID, Client_ID, Patient_ID)",
    "cust_tier": "Customer segment column name (e.g. Loyalty_Status, Wealth_Tier, Insurance_Type)",
    "emp_id": "Employee ID column name (e.g. Employee_ID, Advisor_ID, Rep_ID)",
    "emp_role": "Employee role column name (e.g. Job_Title, Advisor_Level, Sales_Role)",
    "channel": "Transaction channel column name (e.g. Sales_Channel, Admission_Source, Lead_Source)",
    "vol": "Volume metric — what is counted per transaction (e.g. Units_Sold, Procedures_Done, Shipments_Dispatched)",
    "fin": "Financial metric — the currency value per transaction (e.g. Revenue, Billed_Amount, Freight_Value)"
  }},
  "entity_groups": [
    ["<lvl1_value_A>", "<lvl2_value_A>"],
    ["<lvl1_value_B>", "<lvl2_value_B>"],
    ["<lvl1_value_C>", "<lvl2_value_C>"],
    ["<lvl1_value_D>", "<lvl2_value_D>"]
  ],
  "item_groups": [
    ["<item_lvl1_value_A>", "<item_lvl2_value_A>"],
    ["<item_lvl1_value_B>", "<item_lvl2_value_B>"],
    ["<item_lvl1_value_C>", "<item_lvl2_value_C>"],
    ["<item_lvl1_value_D>", "<item_lvl2_value_D>"]
  ],
  "customer_tiers": ["<low_tier>", "<mid_tier>", "<high_tier>"],
  "employee_roles": ["<junior_role>", "<mid_role>", "<senior_role>"],
  "channels": ["<channel_1>", "<channel_2>", "<channel_3>"]
}}

Hard constraints — the output will be rejected if any are violated:
- "entity_groups" must have EXACTLY 4 pairs; each pair must have EXACTLY 2 strings
- "item_groups" must have EXACTLY 4 pairs; each pair must have EXACTLY 2 strings
- "employee_roles" must have EXACTLY 3 entries ordered junior → mid → senior
- All 13 keys under "headers" must be present with non-empty string values
- Replace every placeholder (angle-bracket text) with real {new_ind_name}-specific terminology
- Return ONLY the JSON object — no markdown, no backticks, no explanation"""

        st.code(prompt_text, language="text")
        st.caption(
            "Copy the prompt above into an LLM, then paste its output into the box below."
        )

    st.markdown("**Paste LLM output here:**")
    pasted = st.text_area(
        "LLM output",
        height=200,
        placeholder='{"headers": {...}, "entity_groups": [...], ...}',
        label_visibility="collapsed",
    )

    if st.button("Validate & Add Industry", type="primary"):
        if not new_ind_name.strip():
            st.error("Enter an industry name before validating.")
        elif not pasted.strip():
            st.error("Paste the LLM output before validating.")
        elif new_ind_name.strip() in industries:
            st.error(
                f"'{new_ind_name.strip()}' already exists. Choose a different name."
            )
        else:
            cleaned = pasted.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(lines[1:])
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()

            try:
                parsed = json.loads(cleaned)

                # Handle LLM wrapping the config in {"IndustryName": {...}}
                top_keys = set(parsed.keys())
                expected_keys = {
                    "headers",
                    "entity_groups",
                    "item_groups",
                    "customer_tiers",
                    "employee_roles",
                    "channels",
                }
                if len(parsed) == 1 and not top_keys.intersection(expected_keys):
                    parsed = list(parsed.values())[0]

                errors = _validate_industry_config(parsed)
                if errors:
                    st.error(
                        "Validation failed — fix the following issues and try again:"
                    )
                    for e in errors:
                        st.markdown(f"- {e}")
                else:
                    industries[new_ind_name.strip()] = parsed
                    with open("config.json", "w") as f:
                        json.dump(industries, f, indent=2)
                    load_config.clear()
                    st.session_state["_success_msg"] = f"✅ '{new_ind_name.strip()}' added! Select it from the dropdown."
                    st.rerun()

            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
                st.info(
                    "Make sure you pasted raw JSON only — no surrounding text or markdown code fences."
                )

    # --- MANAGE CUSTOM INDUSTRIES ---
    custom_industries = [k for k in industries if k not in _BUILTIN_INDUSTRIES]
    st.divider()
    st.subheader("⚙️ Manage Custom Industries")

    if not custom_industries:
        st.info("No custom industries yet. Add one above and it will appear here.")
    else:
        mgmt_ind = st.selectbox(
            "Select an industry to manage:",
            options=custom_industries,
            key="mgmt_select",
        )

        rename_col, delete_col = st.columns([2, 1])

        with rename_col:
            with st.container(border=True):
                st.markdown("**Rename**")
                new_name = st.text_input(
                    "New name:", placeholder=f"e.g., {mgmt_ind} v2", key="mgmt_rename_input"
                )
                if st.button("Rename Industry", key="btn_rename"):
                    new_name = new_name.strip()
                    if not new_name:
                        st.error("Enter a new name.")
                    elif new_name in industries:
                        st.error(f"'{new_name}' already exists.")
                    else:
                        industries[new_name] = industries.pop(mgmt_ind)
                        with open("config.json", "w") as f:
                            json.dump(industries, f, indent=2)
                        load_config.clear()
                        st.session_state["_success_msg"] = f"✅ Renamed '{mgmt_ind}' → '{new_name}'."
                        st.rerun()

        with delete_col:
            with st.container(border=True):
                st.markdown("**Delete**")
                confirmed = st.checkbox(
                    "I understand this cannot be undone.", key="del_confirm"
                )
                if st.button(
                    "🗑️ Delete Industry",
                    key="btn_delete",
                    type="secondary",
                    disabled=not confirmed,
                ):
                    del industries[mgmt_ind]
                    with open("config.json", "w") as f:
                        json.dump(industries, f, indent=2)
                    load_config.clear()
                    st.session_state["_success_msg"] = f"🗑️ '{mgmt_ind}' deleted."
                    st.rerun()

    st.stop()


# --- 4. DATA GENERATION ENGINE (CACHED FOR STABILITY) ---
config = industries[selected_ind]
hdrs = config["headers"]


@st.cache_data
def generate_isomorphic_data(ind_name, rows, cfg, seed=42):
    np.random.seed(seed)
    Faker.seed(seed)
    fake = Faker()

    h = cfg["headers"]

    item_modifiers = [
        "Basic",
        "Standard",
        "Advanced",
        "Premium",
        "Enterprise",
        "Essential",
        "Plus",
        "Pro",
        "Ultimate",
        "Signature",
    ]

    # 1. Dim_Entity (~40 rows)
    # local_latlng() returns (latitude, longitude, place_name, country_code, timezone)
    # for real-world locations — enables geospatial map exercises.
    entities = []
    e_id = 101
    for group in cfg["entity_groups"]:
        for _ in range(10):
            place = fake.local_latlng()
            entities.append(
                {
                    "Entity_ID": e_id,
                    h["ent_lvl1"]: group[0],
                    h["ent_lvl2"]: group[1],
                    h["ent_lvl3"]: f"{place[2]} {h['ent_lvl3']}",
                    "City": place[2],
                    "Country_Code": place[3],
                    "Latitude": round(float(place[0]), 4),
                    "Longitude": round(float(place[1]), 4),
                }
            )
            e_id += 1
    dim_entity = pd.DataFrame(entities)

    # 2. Dim_Item (~40 rows)
    items = []
    i_id = 2001
    item_weights = []
    base_prices = {}
    base_volumes = {}

    for group in cfg["item_groups"]:
        for i in range(10):
            weight = np.random.pareto(a=2.0)
            price = int(np.random.uniform(50, 2000))
            items.append(
                {
                    "Item_ID": i_id,
                    h["item_lvl1"]: group[0],
                    h["item_lvl2"]: group[1],
                    h["item_lvl3"]: f"{group[1]} {item_modifiers[i]}",
                }
            )
            item_weights.append(weight)
            base_prices[i_id] = price
            base_volumes[i_id] = weight
            i_id += 1

    dim_item = pd.DataFrame(items)
    item_probs = np.array(item_weights) / sum(item_weights)

    # 3. Dim_Customer (1,000 rows)
    age_brackets = ["18-25", "26-35", "36-50", "51-65", "65+"]
    dim_customer = pd.DataFrame(
        {
            h["cust_id"]: range(10001, 11001),
            "Customer_Name": [fake.name() for _ in range(1000)],
            "Age_Bracket": np.random.choice(
                age_brackets, 1000, p=[0.15, 0.3, 0.3, 0.15, 0.1]
            ),
            h["cust_tier"]: np.random.choice(
                cfg["customer_tiers"], 1000, p=[0.6, 0.3, 0.1]
            ),
        }
    )

    # 4. Dim_Employee (50 rows)
    role_weights = {
        cfg["employee_roles"][0]: 0.5,
        cfg["employee_roles"][1]: 1.0,
        cfg["employee_roles"][2]: 1.5,
    }
    dim_employee = pd.DataFrame(
        {
            h["emp_id"]: range(501, 551),
            "Employee_Name": [fake.name() for _ in range(50)],
            h["emp_role"]: np.random.choice(
                cfg["employee_roles"], 50, p=[0.4, 0.5, 0.1]
            ),
        }
    )
    dim_employee["_perf_weight"] = dim_employee[h["emp_role"]].map(role_weights)

    # 5. Dim_Channel
    dim_channel = pd.DataFrame(
        {
            "Channel_ID": range(1, len(cfg["channels"]) + 1),
            h["channel"]: cfg["channels"],
        }
    )

    # 6. Dim_Date (3 years)
    date_range = pd.date_range(start="2023-01-01", end="2025-12-31")
    dim_date = pd.DataFrame({"Date": date_range})
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Quarter"] = "Q" + dim_date["Date"].dt.quarter.astype(str)
    dim_date["Month_Num"] = dim_date["Date"].dt.month
    dim_date["Month_Name"] = dim_date["Date"].dt.strftime("%b")
    dim_date["Day_of_Week"] = dim_date["Date"].dt.day_name()
    dim_date["Is_Weekend"] = dim_date["Date"].dt.dayofweek >= 5

    # Fact Table
    random_dates = np.random.choice(date_range, rows)
    fact_table = pd.DataFrame(
        {
            "Transaction_ID": range(100000, 100000 + rows),
            "Date": np.sort(random_dates),
            "Entity_ID": np.random.choice(dim_entity["Entity_ID"], rows),
            "Item_ID": np.random.choice(dim_item["Item_ID"], rows, p=item_probs),
            h["cust_id"]: np.random.choice(dim_customer[h["cust_id"]], rows),
            h["emp_id"]: np.random.choice(dim_employee[h["emp_id"]], rows),
            "Channel_ID": np.random.choice(dim_channel["Channel_ID"], rows),
        }
    )

    fact_table["base_p"] = fact_table["Item_ID"].map(base_prices)
    fact_table["base_v"] = fact_table["Item_ID"].map(base_volumes)
    fact_table["emp_w"] = fact_table[h["emp_id"]].map(
        dim_employee.set_index(h["emp_id"])["_perf_weight"]
    )
    fact_table["month"] = fact_table["Date"].dt.month
    fact_table["season_mult"] = np.where(fact_table["month"] >= 11, 1.5, 1.0)

    noise_v = np.random.uniform(0.8, 1.2, rows)
    noise_p = np.random.uniform(0.95, 1.05, rows)

    fact_table[h["vol"]] = np.maximum(
        1,
        fact_table["base_v"]
        * fact_table["emp_w"]
        * fact_table["season_mult"]
        * noise_v
        * 10,
    ).astype(int)
    fact_table[h["fin"]] = (
        fact_table[h["vol"]] * (fact_table["base_p"] * noise_p)
    ).round(2)

    fact_table = fact_table.drop(
        columns=["base_p", "base_v", "emp_w", "month", "season_mult"]
    )
    dim_employee = dim_employee.drop(columns=["_perf_weight"])

    return (
        fact_table,
        dim_date,
        dim_entity,
        dim_item,
        dim_customer,
        dim_employee,
        dim_channel,
    )


f_fact_clean, d_date, d_ent, d_item, d_cust_clean, d_emp, d_chan = (
    generate_isomorphic_data(selected_ind, num_rows, config, seed)
)


# --- 5. DIRTY DATA ---
@st.cache_data
def apply_dirty_data(fact_df, cust_df, vol_col, fin_col, null_pct=3, dup_count=50, outlier_mult=50):
    """
    Injects deterministic dirty data using a separate RNG (seed=99) that does not
    affect the main seed math in generate_isomorphic_data.

    Parameters
    ----------
    null_pct     : % of Customer_Name values to null in Dim_Customer
    dup_count    : number of exact duplicate rows to inject into Fact_Transactions
    outlier_mult : multiplier applied to vol and fin for 20 randomly selected outlier rows
    """
    rng = np.random.default_rng(99)

    # Nulls in Dim_Customer
    cust_dirty = cust_df.copy()
    null_size = max(1, int(len(cust_dirty) * null_pct / 100))
    null_idx = rng.choice(cust_dirty.index, size=null_size, replace=False)
    cust_dirty.loc[null_idx, "Customer_Name"] = None

    fact_dirty = fact_df.copy()

    # Duplicate rows (exact copies, same Transaction_ID)
    if dup_count > 0:
        actual_dups = min(dup_count, len(fact_dirty))
        dup_idx = rng.choice(fact_dirty.index, size=actual_dups, replace=False)
        dups = fact_dirty.loc[dup_idx].copy()
        fact_dirty = pd.concat([fact_dirty, dups]).sort_values("Date").reset_index(drop=True)

    # Outliers: vol and fin inflated by outlier_mult
    outlier_idx = rng.choice(fact_dirty.index, size=20, replace=False)
    fact_dirty.loc[outlier_idx, vol_col] = (
        fact_dirty.loc[outlier_idx, vol_col] * outlier_mult
    ).astype(int)
    fact_dirty.loc[outlier_idx, fin_col] = (
        fact_dirty.loc[outlier_idx, fin_col] * outlier_mult
    ).round(2)

    return fact_dirty, cust_dirty


if dirty_mode:
    null_size = max(1, int(len(d_cust_clean) * null_pct / 100))
    f_fact, d_cust = apply_dirty_data(
        f_fact_clean, d_cust_clean, hdrs["vol"], hdrs["fin"],
        null_pct, dup_count, outlier_mult,
    )
    st.warning(
        "**Dirty Data Mode is active.** Injected issues:\n"
        f"\n- **Nulls:** {null_size} null `Customer_Name` values in `Dim_Customer` ({null_pct}% of rows)"
        f"\n- **Duplicates:** {dup_count} duplicate rows in `Fact_Transactions` (same `Transaction_ID`)"
        f"\n- **Outliers:** 20 rows with volume and financial metrics inflated {outlier_mult}×",
        icon="⚠️",
    )
else:
    f_fact, d_cust = f_fact_clean, d_cust_clean


def _build_schema_dot(h):
    """Return a Graphviz DOT string for the star schema using the given header dict."""

    def _tbl(node_id, title, pk, cols, hc, rc):
        col_rows = "".join(
            f'<TR><TD ALIGN="LEFT" BGCOLOR="{rc}"><FONT POINT-SIZE="9">{c}</FONT></TD></TR>'
            for c in cols
        )
        label = (
            f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
            f'<TR><TD BGCOLOR="{hc}"><B><FONT COLOR="white" POINT-SIZE="10">{title}</FONT></B></TD></TR>'
            f'<TR><TD ALIGN="LEFT" BGCOLOR="{rc}"><B><FONT POINT-SIZE="9">🔑 {pk}</FONT></B></TD></TR>'
            f'{col_rows}'
            f'</TABLE>>'
        )
        return f'{node_id} [label={label} shape=plain margin="0"]'

    FACT_H, FACT_R = "#1a3a5c", "#d6eaf8"
    DIM_H,  DIM_R  = "#1d6a3a", "#d5f5e3"

    nodes = [
        _tbl("Fact", "Fact_Transactions", "Transaction_ID",
             ["Date (FK)", "Entity_ID (FK)", "Item_ID (FK)",
              f"{h['cust_id']} (FK)", f"{h['emp_id']} (FK)", "Channel_ID (FK)",
              h["vol"], h["fin"]],
             FACT_H, FACT_R),
        _tbl("DimDate", "Dim_Date", "Date",
             ["Year", "Quarter", "Month_Num", "Month_Name", "Day_of_Week", "Is_Weekend"],
             DIM_H, DIM_R),
        _tbl("DimEntity", "Dim_Entity", "Entity_ID",
             [h["ent_lvl1"], h["ent_lvl2"], h["ent_lvl3"],
              "City", "Country_Code", "Latitude", "Longitude"],
             DIM_H, DIM_R),
        _tbl("DimItem", "Dim_Item", "Item_ID",
             [h["item_lvl1"], h["item_lvl2"], h["item_lvl3"]],
             DIM_H, DIM_R),
        _tbl("DimCustomer", "Dim_Customer", h["cust_id"],
             ["Customer_Name", "Age_Bracket", h["cust_tier"]],
             DIM_H, DIM_R),
        _tbl("DimEmployee", "Dim_Employee", h["emp_id"],
             ["Employee_Name", h["emp_role"]],
             DIM_H, DIM_R),
        _tbl("DimChannel", "Dim_Channel", "Channel_ID",
             [h["channel"]],
             DIM_H, DIM_R),
    ]

    edges = [
        'Fact -> DimDate     [label=" Date"]',
        'Fact -> DimEntity   [label=" Entity_ID"]',
        'Fact -> DimItem     [label=" Item_ID"]',
        f'Fact -> DimCustomer [label=" {h["cust_id"]}"]',
        f'Fact -> DimEmployee [label=" {h["emp_id"]}"]',
        'Fact -> DimChannel  [label=" Channel_ID"]',
    ]

    body = "\n    ".join(nodes + edges)
    return (
        'digraph StarSchema {\n'
        '    graph [rankdir=LR nodesep=0.7 ranksep=2.5 bgcolor="transparent"]\n'
        '    node  [shape=plain margin="0"]\n'
        '    edge  [color="#555555" fontname="Helvetica" fontsize=9 fontcolor="#333333"]\n'
        f'    {body}\n'
        '}'
    )


# --- 6. CUSTOMIZATION UI (TABS) ---
st.divider()
st.subheader("🛠️ Step 3: Review & Edit Metadata")
st.markdown(
    "Data is populated using `Faker` with a locked seed. "
    "Primary Keys are disabled to protect integrity, but you can edit any text cell below."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Fact_Transactions",
        "Dim_Entity",
        "Dim_Item",
        "Dim_Customer",
        "Dim_Employee",
        "Dim_Channel",
        "Dim_Date",
        "🗂️ Schema Diagram",
    ]
)

with tab1:
    st.write("**Fact Table (Read-Only Preview):**")
    st.dataframe(f_fact.head(50), width="stretch")
with tab2:
    edited_ent = st.data_editor(
        d_ent,
        disabled=["Entity_ID", "Latitude", "Longitude"],
        hide_index=True,
        width="stretch",
    )
with tab3:
    edited_item = st.data_editor(
        d_item, disabled=["Item_ID"], hide_index=True, width="stretch"
    )
with tab4:
    edited_cust = st.data_editor(
        d_cust, disabled=[hdrs["cust_id"]], hide_index=True, width="stretch"
    )
with tab5:
    edited_emp = st.data_editor(
        d_emp, disabled=[hdrs["emp_id"]], hide_index=True, width="stretch"
    )
with tab6:
    edited_chan = st.data_editor(
        d_chan, disabled=["Channel_ID"], hide_index=True, width="stretch"
    )
with tab7:
    st.write("**Dim_Date (Read-Only Preview):**")
    st.dataframe(d_date.head(15), width="stretch")
with tab8:
    st.caption(
        "Star schema for the current industry. Column names update when you switch industries."
    )
    st.graphviz_chart(_build_schema_dot(hdrs))


# --- 7. ANSWER KEY ---
# Compute all tables here (outside the expander) so they are available
# for both the display and the export download button.
vol = hdrs["vol"]
fin = hdrs["fin"]

_ak_dated  = f_fact_clean.merge(d_date[["Date", "Year", "Quarter"]], on="Date")
_ak_entity = f_fact_clean.merge(d_ent, on="Entity_ID")
_ak_item   = f_fact_clean.merge(d_item, on="Item_ID")
_ak_chan   = f_fact_clean.merge(d_chan, on="Channel_ID")
_ak_emp    = f_fact_clean.merge(d_emp, on=hdrs["emp_id"])
_ak_cust   = f_fact_clean.merge(d_cust_clean, on=hdrs["cust_id"])

ak_summary = pd.DataFrame({
    "Metric": [f"Total {fin}", f"Total {vol}", f"Avg {fin} per Transaction", "Total Transactions"],
    "Value":  [
        round(f_fact_clean[fin].sum(), 2),
        int(f_fact_clean[vol].sum()),
        round(f_fact_clean[fin].mean(), 2),
        len(f_fact_clean),
    ],
})
ak_by_year  = _ak_dated.groupby("Year")[fin].sum().reset_index()
ak_by_qtr   = _ak_dated.groupby("Quarter")[fin].sum().reset_index().sort_values("Quarter")
ak_by_chan  = _ak_chan.groupby(hdrs["channel"])[fin].sum().reset_index().sort_values(fin, ascending=False)
ak_by_tier  = _ak_cust.groupby(hdrs["cust_tier"])[fin].sum().reset_index().sort_values(fin, ascending=False)
ak_top_ent  = _ak_entity.groupby(hdrs["ent_lvl3"])[fin].sum().nlargest(5).reset_index()
ak_top_item = _ak_item.groupby(hdrs["item_lvl2"])[fin].sum().nlargest(5).reset_index()
ak_by_role  = _ak_emp.groupby(hdrs["emp_role"])[fin].sum().reset_index().sort_values(fin, ascending=False)

_item_fin   = _ak_item.groupby("Item_ID")[fin].sum().sort_values(ascending=False)
_cumulative = _item_fin.cumsum() / _item_fin.sum()
ak_pareto_n   = int((_cumulative < 0.8).sum()) + 1
ak_pareto_pct = ak_pareto_n / len(_item_fin) * 100
_q4_mask      = _ak_dated["Quarter"] == "Q4"
ak_q4_avg     = _ak_dated.loc[_q4_mask, fin].mean()
ak_non_q4_avg = _ak_dated.loc[~_q4_mask, fin].mean()
ak_validation = pd.DataFrame({
    "Check":  ["Pareto Rule", "Q4 Seasonality"],
    "Result": [
        f"Top {ak_pareto_n}/{len(_item_fin)} items ({ak_pareto_pct:.0f}%) ≈ 80% of {fin}",
        f"Q4 avg {ak_q4_avg:,.2f} vs non-Q4 {ak_non_q4_avg:,.2f} (ratio {ak_q4_avg / ak_non_q4_avg:.2f}x)",
    ],
})

# Build the export ZIP in memory
_ak_zip_buf = io.BytesIO()
with zipfile.ZipFile(_ak_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
    _zf.writestr("AK_Summary.csv",          ak_summary.to_csv(index=False))
    _zf.writestr("AK_By_Year.csv",           ak_by_year.to_csv(index=False))
    _zf.writestr("AK_By_Quarter.csv",        ak_by_qtr.to_csv(index=False))
    _zf.writestr(f"AK_By_{hdrs['channel']}.csv",   ak_by_chan.to_csv(index=False))
    _zf.writestr(f"AK_By_{hdrs['cust_tier']}.csv", ak_by_tier.to_csv(index=False))
    _zf.writestr(f"AK_Top5_{hdrs['ent_lvl3']}.csv", ak_top_ent.to_csv(index=False))
    _zf.writestr(f"AK_Top5_{hdrs['item_lvl2']}.csv", ak_top_item.to_csv(index=False))
    _zf.writestr(f"AK_By_{hdrs['emp_role']}.csv",  ak_by_role.to_csv(index=False))
    _zf.writestr("AK_Validation_Checks.csv", ak_validation.to_csv(index=False))

st.divider()
with st.expander("📋 Answer Key (Instructor Use — Always Based on Clean Data)"):
    st.download_button(
        label="⬇️ Download Answer Key (ZIP of CSVs)",
        data=_ak_zip_buf.getvalue(),
        file_name=f"{selected_ind}_Seed{seed}_Answer_Key.zip",
        mime="application/zip",
    )
    st.divider()

    st.markdown("#### Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Total {fin}",             f"{ak_summary.loc[0, 'Value']:,.0f}")
    m2.metric(f"Total {vol}",             f"{ak_summary.loc[1, 'Value']:,}")
    m3.metric(f"Avg {fin} / Transaction", f"{ak_summary.loc[2, 'Value']:,.2f}")
    m4.metric("Total Transactions",       f"{ak_summary.loc[3, 'Value']:,}")

    st.markdown("#### Breakdown by Dimension")
    ak_left, ak_right = st.columns(2)

    def _fmt(df, col):
        """Return a display copy with the numeric column formatted."""
        out = df.copy()
        out[col] = out[col].map("{:,.0f}".format)
        return out

    with ak_left:
        st.markdown(f"**{fin} by Year**")
        st.dataframe(_fmt(ak_by_year, fin), hide_index=True, width="stretch")

        st.markdown(f"**{fin} by Quarter**")
        st.dataframe(_fmt(ak_by_qtr, fin), hide_index=True, width="stretch")

        st.markdown(f"**{fin} by {hdrs['channel']}**")
        st.dataframe(_fmt(ak_by_chan, fin), hide_index=True, width="stretch")

        st.markdown(f"**{fin} by {hdrs['cust_tier']}**")
        st.dataframe(_fmt(ak_by_tier, fin), hide_index=True, width="stretch")

    with ak_right:
        st.markdown(f"**Top 5 {hdrs['ent_lvl3']} by {fin}**")
        st.dataframe(_fmt(ak_top_ent, fin), hide_index=True, width="stretch")

        st.markdown(f"**Top 5 {hdrs['item_lvl2']} by {fin}**")
        st.dataframe(_fmt(ak_top_item, fin), hide_index=True, width="stretch")

        st.markdown(f"**{fin} by {hdrs['emp_role']}**")
        st.dataframe(_fmt(ak_by_role, fin), hide_index=True, width="stretch")

    st.markdown("#### Validation Checks")
    st.info(
        f"**Pareto:** Top **{ak_pareto_n}** of {len(_item_fin)} items "
        f"({ak_pareto_pct:.0f}% of catalog) account for ~80% of total {fin}."
    )
    st.info(
        f"**Q4 Seasonality:** Avg {fin} per transaction — "
        f"Q4: **{ak_q4_avg:,.2f}** vs. non-Q4: **{ak_non_q4_avg:,.2f}** "
        f"(ratio: {ak_q4_avg / ak_non_q4_avg:.2f}x, expected ~1.5x)"
    )


# --- 8. ZIP EXPORT ---
st.divider()
st.subheader("📦 Step 4: Export Schema")

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
    zip_file.writestr("Fact_Transactions.csv", f_fact.to_csv(index=False))
    zip_file.writestr("Dim_Date.csv", d_date.to_csv(index=False))
    zip_file.writestr("Dim_Entity.csv", edited_ent.to_csv(index=False))
    zip_file.writestr("Dim_Item.csv", edited_item.to_csv(index=False))
    zip_file.writestr("Dim_Customer.csv", edited_cust.to_csv(index=False))
    zip_file.writestr("Dim_Employee.csv", edited_emp.to_csv(index=False))
    zip_file.writestr("Dim_Channel.csv", edited_chan.to_csv(index=False))

st.download_button(
    label=f"Download {selected_ind}{' [DIRTY]' if dirty_mode else ''} Schema (ZIP)",
    data=zip_buffer.getvalue(),
    file_name=f"{selected_ind}{'_DIRTY' if dirty_mode else ''}_MSTR_Schema.zip",
    mime="application/zip",
    type="primary",
)

# --- 9. COHORT BATCH EXPORT ---
st.divider()
with st.expander("🎓 Cohort Batch Export (Instructor Use)"):
    st.markdown(
        "Generate datasets for multiple industries from a single seed in one ZIP. "
        "Each industry becomes its own subfolder — distribute the file to your cohort "
        "and every learner's dataset shares the same underlying math and answer key."
    )

    col_bi, col_br, col_bs = st.columns([3, 1, 1])
    with col_bi:
        batch_inds = st.multiselect(
            "Industries to include:",
            options=list(industries.keys()),
            default=[selected_ind],
            key="batch_inds",
        )
    with col_br:
        batch_rows = st.number_input(
            "Rows per dataset",
            min_value=5000, max_value=50000, value=int(num_rows), step=5000,
            key="batch_rows",
        )
    with col_bs:
        batch_seed = st.number_input(
            "Seed",
            min_value=0, max_value=99999, value=int(seed), step=1,
            key="batch_seed",
        )

    if st.button(
        f"Generate Cohort ZIP  ({len(batch_inds)} {'industry' if len(batch_inds) == 1 else 'industries'})",
        type="primary",
        disabled=len(batch_inds) == 0,
        key="btn_cohort",
    ):
        cohort_buf = io.BytesIO()
        bar = st.progress(0, text="Starting…")
        with zipfile.ZipFile(cohort_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, ind_name in enumerate(batch_inds):
                bar.progress(i / len(batch_inds), text=f"Generating {ind_name}…")
                cfg = industries[ind_name]
                fact, ddate, dent, ditem, dcust, demp, dchan = generate_isomorphic_data(
                    ind_name, int(batch_rows), cfg, int(batch_seed)
                )
                prefix = f"{ind_name}_Seed{int(batch_seed)}/"
                zf.writestr(prefix + "Fact_Transactions.csv", fact.to_csv(index=False))
                zf.writestr(prefix + "Dim_Date.csv",          ddate.to_csv(index=False))
                zf.writestr(prefix + "Dim_Entity.csv",        dent.to_csv(index=False))
                zf.writestr(prefix + "Dim_Item.csv",          ditem.to_csv(index=False))
                zf.writestr(prefix + "Dim_Customer.csv",      dcust.to_csv(index=False))
                zf.writestr(prefix + "Dim_Employee.csv",      demp.to_csv(index=False))
                zf.writestr(prefix + "Dim_Channel.csv",       dchan.to_csv(index=False))
            bar.progress(1.0, text="Done!")
        bar.empty()
        st.session_state["_cohort_zip"]   = cohort_buf.getvalue()
        st.session_state["_cohort_fname"] = (
            f"Cohort_Seed{int(batch_seed)}_{len(batch_inds)}industries.zip"
        )
        st.session_state["_cohort_label"] = (
            f"⬇️ Download Cohort ZIP — {len(batch_inds)} "
            f"{'industry' if len(batch_inds) == 1 else 'industries'}, "
            f"Seed {int(batch_seed)}, {int(batch_rows):,} rows each"
        )

    if "_cohort_zip" in st.session_state:
        st.download_button(
            label=st.session_state["_cohort_label"],
            data=st.session_state["_cohort_zip"],
            file_name=st.session_state["_cohort_fname"],
            mime="application/zip",
            key="dl_cohort",
        )
