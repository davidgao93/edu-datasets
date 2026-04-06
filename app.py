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
with col2:
    if selected_ind != _ADD_SENTINEL:
        dirty_mode = st.checkbox(
            "Enable Dirty Data Mode",
            help=(
                "Injects mathematically locked nulls, duplicates, and outliers "
                "into the export for data cleansing / wrangling exercises."
            ),
        )


# --- 3. ADD YOUR OWN INDUSTRY ---
if selected_ind == _ADD_SENTINEL:
    st.divider()
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
                    st.toast(
                        f"✅ '{new_ind_name.strip()}' added! Select it from the dropdown."
                    )
                    st.rerun()

            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
                st.info(
                    "Make sure you pasted raw JSON only — no surrounding text or markdown code fences."
                )

    st.stop()


# --- 4. DATA GENERATION ENGINE (CACHED FOR STABILITY) ---
config = industries[selected_ind]
hdrs = config["headers"]


@st.cache_data
def generate_isomorphic_data(ind_name, rows, cfg):
    # LOCK THE MATH & STRINGS
    np.random.seed(42)
    Faker.seed(42)
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
    entities = []
    e_id = 101
    for group in cfg["entity_groups"]:
        for _ in range(10):
            city_name = fake.city()
            entities.append(
                {
                    "Entity_ID": e_id,
                    h["ent_lvl1"]: group[0],
                    h["ent_lvl2"]: group[1],
                    h["ent_lvl3"]: f"{city_name} {h['ent_lvl3']}",
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

    # 6. Dim_Date (2 years)
    date_range = pd.date_range(start="2023-01-01", end="2024-12-31")
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
    generate_isomorphic_data(selected_ind, num_rows, config)
)


# --- 5. DIRTY DATA ---
@st.cache_data
def apply_dirty_data(fact_df, cust_df, vol_col, fin_col):
    """
    Injects deterministic dirty data using a separate RNG (seed=99) that does not
    affect the locked seed-42 math in generate_isomorphic_data.

    Injected issues:
      - Nulls:      30 null Customer_Name values in Dim_Customer (~3%)
      - Duplicates: 50 duplicate rows in Fact_Transactions (same Transaction_ID)
      - Outliers:   20 rows in Fact_Transactions with vol and fin inflated 50x
    """
    rng = np.random.default_rng(99)

    # Nulls in Dim_Customer
    cust_dirty = cust_df.copy()
    null_idx = rng.choice(cust_dirty.index, size=30, replace=False)
    cust_dirty.loc[null_idx, "Customer_Name"] = None

    fact_dirty = fact_df.copy()

    # Duplicate rows (exact copies, same Transaction_ID)
    dup_idx = rng.choice(fact_dirty.index, size=50, replace=False)
    dups = fact_dirty.loc[dup_idx].copy()
    fact_dirty = (
        pd.concat([fact_dirty, dups]).sort_values("Date").reset_index(drop=True)
    )

    # Outliers: vol and fin inflated 50x
    outlier_idx = rng.choice(fact_dirty.index, size=20, replace=False)
    fact_dirty.loc[outlier_idx, vol_col] = (
        fact_dirty.loc[outlier_idx, vol_col] * 50
    ).astype(int)
    fact_dirty.loc[outlier_idx, fin_col] = (
        fact_dirty.loc[outlier_idx, fin_col] * 50
    ).round(2)

    return fact_dirty, cust_dirty


_DIRTY_MANIFEST = [
    ("Nulls", "30 null `Customer_Name` values in `Dim_Customer` (~3% of rows)"),
    (
        "Duplicates",
        "50 duplicate rows in `Fact_Transactions` (exact copies, same `Transaction_ID`)",
    ),
    (
        "Outliers",
        "20 rows in `Fact_Transactions` with volume and financial metrics inflated 50x",
    ),
]

if dirty_mode:
    f_fact, d_cust = apply_dirty_data(
        f_fact_clean, d_cust_clean, hdrs["vol"], hdrs["fin"]
    )
    st.warning(
        "**Dirty Data Mode is active.** The exported dataset contains the following injected issues:\n"
        + "".join(f"\n- **{label}:** {desc}" for label, desc in _DIRTY_MANIFEST),
        icon="⚠️",
    )
else:
    f_fact, d_cust = f_fact_clean, d_cust_clean


# --- 6. CUSTOMIZATION UI (TABS) ---
st.divider()
st.subheader("🛠️ Step 3: Review & Edit Metadata")
st.markdown(
    "Data is populated using `Faker` with a locked seed. "
    "Primary Keys are disabled to protect integrity, but you can edit any text cell below."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Fact_Transactions",
        "Dim_Entity",
        "Dim_Item",
        "Dim_Customer",
        "Dim_Employee",
        "Dim_Channel",
        "Dim_Date",
    ]
)

with tab1:
    st.write("**Fact Table (Read-Only Preview):**")
    st.dataframe(f_fact.head(50), width="stretch")
with tab2:
    edited_ent = st.data_editor(
        d_ent, disabled=["Entity_ID"], hide_index=True, width="stretch"
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


# --- 7. ANSWER KEY ---
st.divider()
with st.expander("📋 Answer Key (Instructor Use — Always Based on Clean Data)"):
    vol = hdrs["vol"]
    fin = hdrs["fin"]

    # Pre-join clean tables for analysis
    fact_dated = f_fact_clean.merge(d_date[["Date", "Year", "Quarter"]], on="Date")
    fact_entity = f_fact_clean.merge(d_ent, on="Entity_ID")
    fact_item = f_fact_clean.merge(d_item, on="Item_ID")
    fact_chan = f_fact_clean.merge(d_chan, on="Channel_ID")
    fact_emp = f_fact_clean.merge(d_emp, on=hdrs["emp_id"])
    fact_cust = f_fact_clean.merge(d_cust_clean, on=hdrs["cust_id"])

    # Summary metrics
    total_fin = f_fact_clean[fin].sum()
    total_vol = int(f_fact_clean[vol].sum())
    avg_fin_txn = f_fact_clean[fin].mean()
    n_txns = len(f_fact_clean)

    st.markdown("#### Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Total {fin}", f"{total_fin:,.0f}")
    m2.metric(f"Total {vol}", f"{total_vol:,}")
    m3.metric(f"Avg {fin} / Transaction", f"{avg_fin_txn:,.2f}")
    m4.metric("Total Transactions", f"{n_txns:,}")

    st.markdown("#### Breakdown by Dimension")
    ak_left, ak_right = st.columns(2)

    with ak_left:
        st.markdown(f"**{fin} by Year**")
        by_year = fact_dated.groupby("Year")[fin].sum().reset_index()
        by_year[fin] = by_year[fin].map("{:,.0f}".format)
        st.dataframe(by_year, hide_index=True, width="stretch")

        st.markdown(f"**{fin} by Quarter**")
        by_qtr = (
            fact_dated.groupby("Quarter")[fin]
            .sum()
            .reset_index()
            .sort_values("Quarter")
        )
        by_qtr[fin] = by_qtr[fin].map("{:,.0f}".format)
        st.dataframe(by_qtr, hide_index=True, width="stretch")

        st.markdown(f"**{fin} by {hdrs['channel']}**")
        by_chan = (
            fact_chan.groupby(hdrs["channel"])[fin]
            .sum()
            .reset_index()
            .sort_values(fin, ascending=False)
        )
        by_chan[fin] = by_chan[fin].map("{:,.0f}".format)
        st.dataframe(by_chan, hide_index=True, width="stretch")

        st.markdown(f"**{fin} by {hdrs['cust_tier']}**")
        by_tier = (
            fact_cust.groupby(hdrs["cust_tier"])[fin]
            .sum()
            .reset_index()
            .sort_values(fin, ascending=False)
        )
        by_tier[fin] = by_tier[fin].map("{:,.0f}".format)
        st.dataframe(by_tier, hide_index=True, width="stretch")

    with ak_right:
        st.markdown(f"**Top 5 {hdrs['ent_lvl3']} by {fin}**")
        top_ent = (
            fact_entity.groupby(hdrs["ent_lvl3"])[fin].sum().nlargest(5).reset_index()
        )
        top_ent[fin] = top_ent[fin].map("{:,.0f}".format)
        st.dataframe(top_ent, hide_index=True, width="stretch")

        st.markdown(f"**Top 5 {hdrs['item_lvl2']} by {fin}**")
        top_item = (
            fact_item.groupby(hdrs["item_lvl2"])[fin].sum().nlargest(5).reset_index()
        )
        top_item[fin] = top_item[fin].map("{:,.0f}".format)
        st.dataframe(top_item, hide_index=True, width="stretch")

        st.markdown(f"**{fin} by {hdrs['emp_role']}**")
        by_role = (
            fact_emp.groupby(hdrs["emp_role"])[fin]
            .sum()
            .reset_index()
            .sort_values(fin, ascending=False)
        )
        by_role[fin] = by_role[fin].map("{:,.0f}".format)
        st.dataframe(by_role, hide_index=True, width="stretch")

    st.markdown("#### Validation Checks")

    # Pareto insight
    item_fin = fact_item.groupby("Item_ID")[fin].sum().sort_values(ascending=False)
    cumulative = item_fin.cumsum() / item_fin.sum()
    pareto_n = int((cumulative < 0.8).sum()) + 1
    pareto_pct = pareto_n / len(item_fin) * 100
    st.info(
        f"**Pareto:** Top **{pareto_n}** of {len(item_fin)} items "
        f"({pareto_pct:.0f}% of catalog) account for ~80% of total {fin}."
    )

    # Q4 seasonality confirmation
    q4_mask = fact_dated["Quarter"] == "Q4"
    q4_avg = fact_dated.loc[q4_mask, fin].mean()
    non_q4_avg = fact_dated.loc[~q4_mask, fin].mean()
    st.info(
        f"**Q4 Seasonality:** Avg {fin} per transaction — "
        f"Q4: **{q4_avg:,.2f}** vs. non-Q4: **{non_q4_avg:,.2f}** "
        f"(ratio: {q4_avg / non_q4_avg:.2f}x, expected ~1.5x)"
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
