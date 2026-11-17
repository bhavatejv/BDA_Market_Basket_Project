import streamlit as st
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
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    color: #d1d5db;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.metric-title {
    font-size: 14px;
    color: #6b7280;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.insight {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #2563eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #6b7280;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🛒 Market Basket Intelligence</h1>

<p>
Big Data Analytics • Association Rule Mining • Apache Spark
</p>

<p>
Discover hidden relationships between products and generate
data-driven recommendations from transaction data.
</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Dashboard Controls")

    st.markdown("### Data Sources")

    frequent_path = "output/frequent_itemsets"
    rules_path = "output/association_rules"

    st.info(
        "The dashboard automatically searches the Spark output "
        "folders for CSV result files."
    )

    st.markdown("---")

    st.markdown("### 🔬 Technology Stack")

    st.markdown("""
    **Big Data**
    - Apache Spark
    - PySpark

    **Analytics**
    - Association Rule Mining
    - Frequent Itemset Mining

    **Visualization**
    - Streamlit
    - Pandas

    **Language**
    - Python
    """)

    st.markdown("---")

    st.markdown("### 📌 Key Metrics")

    st.write("• Support")
    st.write("• Confidence")
    st.write("• Lift")

    st.markdown("---")

    st.caption("BDA Academic Project")
    st.caption("Market Basket Analysis")


# ---------------------------------------------------------
# FIND CSV FILES
# ---------------------------------------------------------

def find_csv(folder):

    files = []

    if os.path.exists(folder):

        for root, dirs, filenames in os.walk(folder):

            for filename in filenames:

                if filename.lower().endswith(".csv"):

                    files.append(
                        os.path.join(root, filename)
                    )

    return files


frequent_files = find_csv(frequent_path)
rules_files = find_csv(rules_path)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df_frequent = pd.DataFrame()
df_rules = pd.DataFrame()


if frequent_files:

    try:

        df_frequent = pd.read_csv(
            frequent_files[0]
        )

    except Exception as e:

        st.error(
            f"Unable to read frequent itemsets: {e}"
        )


if rules_files:

    try:

        df_rules = pd.read_csv(
            rules_files[0]
        )

    except Exception as e:

        st.error(
            f"Unable to read association rules: {e}"
        )


# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Project Overview</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        📦 Frequent Itemsets
        </div>

        <div class="metric-value">
        {len(df_frequent)}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        🔗 Association Rules
        </div>

        <div class="metric-value">
        {len(df_rules)}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    if not df_frequent.empty:

        freq_columns = [
            c for c in df_frequent.columns
            if c.lower() in ["freq", "frequency", "count"]
        ]

        if freq_columns:

            value = df_frequent[
                freq_columns[0]
            ].max()

        else:

            value = "—"

    else:

        value = "—"


    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        📈 Maximum Frequency
        </div>

        <div class="metric-value">
        {value}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    if not df_rules.empty:

        confidence_columns = [
            c for c in df_rules.columns
            if "confidence" in c.lower()
        ]

        if confidence_columns:

            confidence = df_rules[
                confidence_columns[0]
            ].max()

            try:

                confidence = f"{float(confidence):.2f}"

            except:

                pass

        else:

            confidence = "—"

    else:

        confidence = "—"


    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        🎯 Best Confidence
        </div>

        <div class="metric-value">
        {confidence}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# DATA STATUS
# ---------------------------------------------------------

st.markdown("---")

if df_frequent.empty and df_rules.empty:

    st.warning(
        "⚠️ Spark results have not been generated yet. "
        "Run the Market Basket Spark program first."
    )

else:

    st.success(
        "✅ Spark analysis results successfully loaded."
    )


# ---------------------------------------------------------
# FREQUENT ITEMSETS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📦 Frequent Itemset Analysis</div>',
    unsafe_allow_html=True
)


if not df_frequent.empty:

    st.write(
        "Frequent itemsets represent combinations of products "
        "that appear together frequently in customer transactions."
    )

    # Identify frequency column

    freq_columns = [
        c for c in df_frequent.columns
        if c.lower() in ["freq", "frequency", "count"]
    ]

    if freq_columns:

        freq_col = freq_columns[0]

        top_frequency = (
            df_frequent
            .sort_values(
                freq_col,
                ascending=False
            )
            .head(10)
        )

        chart_data = top_frequency[
            [freq_col]
        ]

        st.bar_chart(
            chart_data
        )


    st.subheader("🔎 Filter Frequent Itemsets")

    search = st.text_input(
        "Search itemset",
        placeholder="Example: Milk, Bread..."
    )


    filtered = df_frequent.copy()


    if search:

        mask = filtered.astype(
            str
        ).apply(
            lambda row:
            row.str.contains(
                search,
                case=False,
                na=False
            ).any(),
            axis=1
        )

        filtered = filtered[mask]


    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )


    st.download_button(
        "⬇️ Download Frequent Itemsets",
        filtered.to_csv(index=False),
        "frequent_itemsets.csv",
        "text/csv"
    )

else:

    st.info(
        "No frequent itemset results available."
    )


# ---------------------------------------------------------
# ASSOCIATION RULES
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🔗 Association Rule Analysis</div>',
    unsafe_allow_html=True
)


if not df_rules.empty:

    st.write(
        "Association rules identify relationships between products "
        "and help discover potential cross-selling opportunities."
    )


    # Detect important columns

    confidence_columns = [
        c for c in df_rules.columns
        if "confidence" in c.lower()
    ]

    lift_columns = [
        c for c in df_rules.columns
        if "lift" in c.lower()
    ]

    support_columns = [
        c for c in df_rules.columns
        if "support" in c.lower()
    ]


    # Rule metrics

    r1, r2, r3 = st.columns(3)


    with r1:

        if support_columns:

            support_value = pd.to_numeric(
                df_rules[
                    support_columns[0]
                ],
                errors="coerce"
            ).max()

            st.metric(
                "📊 Maximum Support",
                f"{support_value:.3f}"
                if pd.notna(support_value)
                else "—"
            )

        else:

            st.metric(
                "📊 Maximum Support",
                "—"
            )


    with r2:

        if confidence_columns:

            confidence_value = pd.to_numeric(
                df_rules[
                    confidence_columns[0]
                ],
                errors="coerce"
            ).max()

            st.metric(
                "🎯 Maximum Confidence",
                f"{confidence_value:.3f}"
                if pd.notna(confidence_value)
                else "—"
            )

        else:

            st.metric(
                "🎯 Maximum Confidence",
                "—"
            )


    with r3:

        if lift_columns:

            lift_value = pd.to_numeric(
                df_rules[
                    lift_columns[0]
                ],
                errors="coerce"
            ).max()

            st.metric(
                "🚀 Maximum Lift",
                f"{lift_value:.3f}"
                if pd.notna(lift_value)
                else "—"
            )

        else:

            st.metric(
                "🚀 Maximum Lift",
                "—"
            )


    # Confidence chart

    if confidence_columns:

        st.subheader(
            "🎯 Top Association Rules by Confidence"
        )

        chart_rules = df_rules.copy()

        chart_rules[
            confidence_columns[0]
        ] = pd.to_numeric(
            chart_rules[
                confidence_columns[0]
            ],
            errors="coerce"
        )

        chart_rules = (
            chart_rules
            .sort_values(
                confidence_columns[0],
                ascending=False
            )
            .head(10)
        )


        st.bar_chart(
            chart_rules[
                [confidence_columns[0]]
            ]
        )


    # Search rules

    st.subheader("🔎 Search Association Rules")

    rule_search = st.text_input(
        "Search product/rule",
        placeholder="Example: Milk"
    )


    filtered_rules = df_rules.copy()


    if rule_search:

        mask = filtered_rules.astype(
            str
        ).apply(
            lambda row:
            row.str.contains(
                rule_search,
                case=False,
                na=False
            ).any(),
            axis=1
        )

        filtered_rules = filtered_rules[
            mask
        ]


    st.dataframe(
        filtered_rules,
        use_container_width=True,
        hide_index=True
    )


    st.download_button(
        "⬇️ Download Association Rules",
        filtered_rules.to_csv(index=False),
        "association_rules.csv",
        "text/csv"
    )


else:

    st.info(
        "No association rule results available."
    )


# ---------------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">💡 Business Insights</div>',
    unsafe_allow_html=True
)


if not df_rules.empty:

    if confidence_columns:

        best_rule = df_rules.loc[
            pd.to_numeric(
                df_rules[
                    confidence_columns[0]
                ],
                errors="coerce"
            ).idxmax()
        ]

        st.markdown(
            """
            <div class="insight">

            <b>🎯 Recommendation Strength</b><br>

            The rule with the highest confidence represents
            one of the strongest purchasing relationships in
            the dataset.

            </div>
            """,
            unsafe_allow_html=True
        )


    if lift_columns:

        lift_numeric = pd.to_numeric(
            df_rules[
                lift_columns[0]
            ],
            errors="coerce"
        )

        if not lift_numeric.empty:

            best_lift = lift_numeric.max()

            if best_lift > 1:

                st.markdown(
                    f"""
                    <div class="insight">

                    <b>🚀 Strong Product Association</b><br>

                    The highest lift value is
                    <b>{best_lift:.2f}</b>.

                    A lift greater than 1 indicates a positive
                    association between the products.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


st.markdown(
    """
    <div class="insight">

    <b>🛍️ Business Applications</b><br>

    • Product recommendation systems<br>
    • Cross-selling strategies<br>
    • Store shelf optimization<br>
    • Promotional bundle creation<br>
    • E-commerce recommendations<br>
    • Customer purchase behavior analysis

    </div>
    """,
    unsafe_allow_html=True
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

    st.info(
        """
        **1️⃣ Data Collection**

        Transaction data is collected
        from the selected dataset.
        """
    )


with m2:

    st.info(
        """
        **2️⃣ Data Processing**

        Apache Spark processes
        large-scale transaction data.
        """
    )


with m3:

    st.info(
        """
        **3️⃣ Pattern Mining**

        Frequent itemsets and
        association rules are generated.
        """
    )


with m4:

    st.info(
        """
        **4️⃣ Insights**

        Results are converted into
        useful business recommendations.
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🛒 <b>Market Basket Intelligence</b><br>

    Big Data Analytics Project • Apache Spark • PySpark • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)


