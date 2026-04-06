import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker  # <-- NEW: Import Faker

st.set_page_config(page_title="BI Dataset Generator V4", layout="wide")
st.title("📊 Custom BI Dataset Generator V4 (MicroStrategy Optimized)")
st.markdown("Isomorphic Data Architecture with Pareto Distributions and Faker Realism.")


# --- 1. LOAD EXTERNAL CONFIGURATION ---
@st.cache_data
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)


industries = load_config()

# --- 2. TOP BAR CONTROLS ---
col1, col2 = st.columns([1, 2])
with col1:
    selected_ind = st.selectbox(
        "1. Select Industry Context:", options=list(industries.keys())
    )
    num_rows = st.slider(
        "2. Number of Fact Rows:",
        min_value=5000,
        max_value=50000,
        value=10000,
        step=5000,
    )

config = industries[selected_ind]
hdrs = config["headers"]


# --- 3. DATA GENERATION ENGINE (CACHED FOR STABILITY) ---
@st.cache_data
def generate_isomorphic_data(ind_name, rows, cfg):
    # LOCK THE MATH & STRINGS
    np.random.seed(42)
    Faker.seed(42)
    fake = Faker()

    # Universal realistic tiers for products/services
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

    # ---------------------------------------------------------
    # DIMENSIONS
    # ---------------------------------------------------------

    # 1. Dim_Entity (~40 rows) - Using Faker Cities + Industry Header
    entities = []
    e_id = 101
    for group in cfg["entity_groups"]:
        for i in range(10):
            city_name = fake.city()
            entities.append(
                {
                    "Entity_ID": e_id,
                    hdrs["ent_lvl1"]: group[0],
                    hdrs["ent_lvl2"]: group[1],
                    hdrs[
                        "ent_lvl3"
                    ]: f"{city_name} {hdrs['ent_lvl3']}",  # e.g., "Springfield Store" or "Springfield Clinic"
                }
            )
            e_id += 1
    dim_entity = pd.DataFrame(entities)

    # 2. Dim_Item (~40 rows) - Using Tier Modifiers
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
                    hdrs["item_lvl1"]: group[0],
                    hdrs["item_lvl2"]: group[1],
                    hdrs[
                        "item_lvl3"
                    ]: f"{group[1]} {item_modifiers[i]}",  # e.g., "MRI Premium" or "Computers Premium"
                }
            )
            item_weights.append(weight)
            base_prices[i_id] = price
            base_volumes[i_id] = weight
            i_id += 1

    dim_item = pd.DataFrame(items)
    item_probs = np.array(item_weights) / sum(item_weights)

    # 3. Dim_Customer (1,000 rows) - Using Faker Names
    age_brackets = ["18-25", "26-35", "36-50", "51-65", "65+"]
    dim_customer = pd.DataFrame(
        {
            hdrs["cust_id"]: range(10001, 11001),
            "Customer_Name": [
                fake.name() for _ in range(1000)
            ],  # <-- Faker applied here
            "Age_Bracket": np.random.choice(
                age_brackets, 1000, p=[0.15, 0.3, 0.3, 0.15, 0.1]
            ),
            hdrs["cust_tier"]: np.random.choice(
                cfg["customer_tiers"], 1000, p=[0.6, 0.3, 0.1]
            ),
        }
    )

    # 4. Dim_Employee (50 rows) - Using Faker Names
    role_weights = {
        cfg["employee_roles"][0]: 0.5,
        cfg["employee_roles"][1]: 1.0,
        cfg["employee_roles"][2]: 1.5,
    }
    dim_employee = pd.DataFrame(
        {
            hdrs["emp_id"]: range(501, 551),
            "Employee_Name": [fake.name() for _ in range(50)],  # <-- Faker applied here
            hdrs["emp_role"]: np.random.choice(
                cfg["employee_roles"], 50, p=[0.4, 0.5, 0.1]
            ),
        }
    )
    dim_employee["_perf_weight"] = dim_employee[hdrs["emp_role"]].map(role_weights)

    # 5. Dim_Channel
    dim_channel = pd.DataFrame(
        {
            "Channel_ID": range(1, len(cfg["channels"]) + 1),
            hdrs["channel"]: cfg["channels"],
        }
    )

    # 6. Dim_Date (2 Years)
    date_range = pd.date_range(start="2023-01-01", end="2024-12-31")
    dim_date = pd.DataFrame({"Date": date_range})
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Quarter"] = "Q" + dim_date["Date"].dt.quarter.astype(str)
    dim_date["Month_Num"] = dim_date["Date"].dt.month
    dim_date["Month_Name"] = dim_date["Date"].dt.strftime("%b")
    dim_date["Day_of_Week"] = dim_date["Date"].dt.day_name()
    dim_date["Is_Weekend"] = dim_date["Date"].dt.dayofweek >= 5

    # ---------------------------------------------------------
    # FACT TABLE
    # ---------------------------------------------------------
    random_dates = np.random.choice(date_range, rows)
    fact_table = pd.DataFrame(
        {
            "Transaction_ID": range(100000, 100000 + rows),
            "Date": np.sort(random_dates),
            "Entity_ID": np.random.choice(dim_entity["Entity_ID"], rows),
            "Item_ID": np.random.choice(dim_item["Item_ID"], rows, p=item_probs),
            hdrs["cust_id"]: np.random.choice(dim_customer[hdrs["cust_id"]], rows),
            hdrs["emp_id"]: np.random.choice(dim_employee[hdrs["emp_id"]], rows),
            "Channel_ID": np.random.choice(dim_channel["Channel_ID"], rows),
        }
    )

    # Math Application
    fact_table["base_p"] = fact_table["Item_ID"].map(base_prices)
    fact_table["base_v"] = fact_table["Item_ID"].map(base_volumes)
    fact_table["emp_w"] = fact_table[hdrs["emp_id"]].map(
        dim_employee.set_index(hdrs["emp_id"])["_perf_weight"]
    )

    # Q4 Seasonality
    fact_table["month"] = fact_table["Date"].dt.month
    fact_table["season_mult"] = np.where(fact_table["month"] >= 11, 1.5, 1.0)

    # Generate final metrics
    noise_v = np.random.uniform(0.8, 1.2, rows)
    noise_p = np.random.uniform(0.95, 1.05, rows)

    fact_table[hdrs["vol"]] = np.maximum(
        1,
        (
            fact_table["base_v"]
            * fact_table["emp_w"]
            * fact_table["season_mult"]
            * noise_v
            * 10
        ),
    ).astype(int)
    fact_table[hdrs["fin"]] = (
        fact_table[hdrs["vol"]] * (fact_table["base_p"] * noise_p)
    ).round(2)

    # Cleanup
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


f_fact, d_date, d_ent, d_item, d_cust, d_emp, d_chan = generate_isomorphic_data(
    selected_ind, num_rows, config
)

# --- 4. CUSTOMIZATION UI (TABS) ---
st.divider()
st.subheader("🛠️ Step 3: Review & Edit Metadata")
st.markdown(
    "Data is populated using `Faker` with a locked seed. Primary Keys are disabled to protect integrity, but you can edit any text cell below."
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
    st.dataframe(f_fact.head(50), use_container_width=True)
with tab2:
    edited_ent = st.data_editor(
        d_ent, disabled=["Entity_ID"], hide_index=True, use_container_width=True
    )
with tab3:
    edited_item = st.data_editor(
        d_item, disabled=["Item_ID"], hide_index=True, use_container_width=True
    )
with tab4:
    edited_cust = st.data_editor(
        d_cust, disabled=[hdrs["cust_id"]], hide_index=True, use_container_width=True
    )
with tab5:
    edited_emp = st.data_editor(
        d_emp, disabled=[hdrs["emp_id"]], hide_index=True, use_container_width=True
    )
with tab6:
    edited_chan = st.data_editor(
        d_chan, disabled=["Channel_ID"], hide_index=True, use_container_width=True
    )
with tab7:
    st.write("**Dim_Date (Read-Only Preview):**")
    st.dataframe(d_date.head(15), use_container_width=True)

# --- 5. ZIP EXPORT ---
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
    label=f"Download {selected_ind} Schema (ZIP)",
    data=zip_buffer.getvalue(),
    file_name=f"{selected_ind}_MSTR_Schema.zip",
    mime="application/zip",
    type="primary",
)
