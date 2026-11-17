import streamlit as st
import pandas as pd
import os
import ast

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Market Basket Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ITEMSET_FILE = os.path.join(
    BASE_DIR,
    "results_spark",
    "fp_growth_itemsets",
    "part-00000.csv"
)

RULE_FILE = os.path.join(
    BASE_DIR,
    "results_spark",
    "fp_growth_rules",
    "part-00000.csv"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}

.metric-title {
    font-size: 15px;
    color: #666;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
}

.section-title {
    font-size: 26px;
    font-weight: 650;
    margin-top: 30px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🛒 Market Basket Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Big Data Analytics using Apache Spark, FP-Growth and Association Rule Mining</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CHECK RESULTS
# ---------------------------------------------------------

if not os.path.exists(ITEMSET_FILE) or not os.path.exists(RULE_FILE):

    st.error("Spark results have not been generated yet.")

    st.write("Expected files:")

    st.code(
        f"{ITEMSET_FILE}\n{RULE_FILE}"
    )

    st.info(
        "Run market_basket_spark_project.py first."
    )

    st.stop()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:

    df_frequent = pd.read_csv(ITEMSET_FILE)
    df_rules = pd.read_csv(RULE_FILE)

except Exception as e:

    st.error("Unable to load Spark result files.")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

def parse_items(value):

    try:
        return ast.literal_eval(value)
    except:
        return [str(value)]


if "items" in df_frequent.columns:
    df_frequent["items_list"] = df_frequent["items"].apply(parse_items)
    df_frequent["Product"] = df_frequent["items_list"].apply(
        lambda x: ", ".join(x)
    )

if "antecedent" in df_rules.columns:
    df_rules["antecedent_list"] = df_rules["antecedent"].apply(parse_items)

if "consequent" in df_rules.columns:
    df_rules["consequent_list"] = df_rules["consequent"].apply(parse_items)

if "antecedent" in df_rules.columns and "consequent" in df_rules.columns:

    df_rules["Rule"] = (
        df_rules["antecedent_list"].apply(lambda x: ", ".join(x))
        + " → "
        + df_rules["consequent_list"].apply(lambda x: ", ".join(x))
    )

# ---------------------------------------------------------
# KEY METRICS
# ---------------------------------------------------------

frequent_itemsets_count = len(df_frequent)

rules_count = len(df_rules)

max_frequency = (
    int(df_frequent["freq"].max())
    if "freq" in df_frequent.columns and len(df_frequent) > 0
    else 0
)

best_confidence = (
    float(df_rules["confidence"].max()) * 100
    if "confidence" in df_rules.columns and len(df_rules) > 0
    else 0
)

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Key Metrics</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Frequent Itemsets",
        frequent_itemsets_count
    )

with c2:
    st.metric(
        "Association Rules",
        rules_count
    )

with c3:
    st.metric(
        "Maximum Frequency",
        f"{max_frequency:,}"
    )

with c4:
    st.metric(
        "Best Confidence",
        f"{best_confidence:.2f}%"
    )

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("### Data Sources")

st.sidebar.success("Spark results loaded successfully")

st.sidebar.markdown("### Technology Stack")

st.sidebar.write("🔹 Apache Spark")
st.sidebar.write("🔹 PySpark")
st.sidebar.write("🔹 FP-Growth")
st.sidebar.write("🔹 Association Rule Mining")
st.sidebar.write("🔹 Streamlit")
st.sidebar.write("🔹 Pandas")

# ---------------------------------------------------------
# FREQUENT ITEMSETS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🔬 Frequent Itemset Analysis</div>',
    unsafe_allow_html=True
)

if len(df_frequent) > 0:

    display_items = df_frequent.copy()

    if "Product" in display_items.columns:
        display_items = display_items[
            ["Product", "freq"]
        ].rename(
            columns={
                "Product": "Itemset",
                "freq": "Frequency"
            }
        )

    st.dataframe(
        display_items,
        width="stretch",
        hide_index=True
    )

    st.markdown("### Top Frequent Itemsets")

    chart_data = display_items.head(10).set_index("Itemset")

    st.bar_chart(chart_data["Frequency"])

else:

    st.warning("No frequent itemsets found.")

# ---------------------------------------------------------
# ASSOCIATION RULES
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🔗 Association Rule Analysis</div>',
    unsafe_allow_html=True
)

if len(df_rules) > 0:

    rule_display = df_rules.copy()

    if "Rule" in rule_display.columns:

        rule_display = rule_display[
            [
                "Rule",
                "support",
                "confidence",
                "lift"
            ]
        ]

    rule_display = rule_display.sort_values(
        by="confidence",
        ascending=False
    )

    st.dataframe(
        rule_display,
        width="stretch",
        hide_index=True
    )

else:

    st.warning("No association rules found.")

# ---------------------------------------------------------
# TOP RULES
# ---------------------------------------------------------

if len(df_rules) > 0:

    st.markdown("### Top Association Rules")

    top_rules = df_rules.sort_values(
        by="confidence",
        ascending=False
    ).head(10)

    chart_data = top_rules.copy()

    if "Rule" in chart_data.columns:

        chart_data = chart_data[
            ["Rule", "confidence"]
        ]

        chart_data["confidence"] = (
            chart_data["confidence"] * 100
        )

        chart_data = chart_data.set_index("Rule")

        st.bar_chart(chart_data["confidence"])

# ---------------------------------------------------------
# BUSINESS APPLICATIONS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">💼 Business Applications</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "🛍️ **Product Bundling**\n\n"
        "Identify products that are frequently purchased together."
    )

with col2:
    st.info(
        "🎯 **Cross Selling**\n\n"
        "Recommend related products based on customer purchasing patterns."
    )

with col3:
    st.info(
        "📦 **Inventory Planning**\n\n"
        "Use frequent purchasing patterns to improve stock planning."
    )

# ---------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🧠 Project Methodology</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("### 1️⃣ Data Processing")
    st.write(
        "Cleaned transaction data and removed invalid records."
    )

with m2:
    st.markdown("### 2️⃣ Frequent Itemsets")
    st.write(
        "FP-Growth was used to discover frequently purchased item combinations."
    )

with m3:
    st.markdown("### 3️⃣ Association Rules")
    st.write(
        "Association rules were generated using support, confidence and lift."
    )

with m4:
    st.markdown("### 4️⃣ Visualization")
    st.write(
        "Streamlit provides an interactive dashboard for analysis."
    )

# ---------------------------------------------------------
# METRICS EXPLANATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📌 Important Metrics</div>',
    unsafe_allow_html=True
)

st.write(
    "**Support:** Measures how frequently an itemset occurs in transactions."
)

st.write(
    "**Confidence:** Measures how often the consequent occurs when the antecedent occurs."
)

st.write(
    "**Lift:** Measures how strongly two products are associated compared with random occurrence."
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        BDA Academic Project | Market Basket Analysis<br>
        Apache Spark • PySpark • FP-Growth • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
