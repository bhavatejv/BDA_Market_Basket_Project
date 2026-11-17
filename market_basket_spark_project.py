
"""
BDA CSE557 - Market Basket Analysis for Retail Sales Optimization

Based on:
P. Kumar, K.N. Manisha, M. Nivetha,
"Market Basket Analysis for Retail Sales Optimization",
ICETITE 2024,
DOI: 10.1109/IC-ETITE58242.2024.10493283.

Run:
    py -m pip install pyspark pandas numpy matplotlib

    py market_basket_spark_project.py

Default input:
    data/simulated_retail_transactions.csv

Output:
    results_spark/
"""
import csv
import json
import shutil
import os
import sys
import time
from itertools import combinations

# ============================================================
# 1. PYTHON AND SPARK CONFIGURATION FOR WINDOWS
# ============================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.fpm import FPGrowth


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

INPUT_PATH = os.environ.get(
    "INPUT_PATH",
    "data/simulated_retail_transactions.csv"
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "results_spark"
)

MIN_SUPPORT = float(
    os.environ.get("MIN_SUPPORT", "0.05")
)

MIN_CONFIDENCE = float(
    os.environ.get("MIN_CONFIDENCE", "0.40")
)


# ============================================================
# 3. CREATE SPARK SESSION
# ============================================================

print("\n============================================")
print(" MARKET BASKET ANALYSIS - SPARK PROJECT")
print("============================================\n")

print("Python executable:", sys.executable)
print("Python version:", sys.version)

spark = (
    SparkSession.builder
    .appName("CSE557-Market-Basket-Analysis")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.python.worker.reuse", "true")
    .config("spark.network.timeout", "300s")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark version:", spark.version)
print("Spark session created successfully!")


try:

    # ========================================================
    # 4. DATA INGESTION
    # ========================================================

    print("\n========== DATA INGESTION ==========\n")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input CSV file not found: {INPUT_PATH}\n"
            "Please check your data folder and INPUT_PATH."
        )

    raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(INPUT_PATH)
    )

    print("Raw schema:")
    raw.printSchema()

    raw_count = raw.count()

    print("Total raw rows:", raw_count)


    # ========================================================
    # 5. DATA CLEANING AND PREPROCESSING
    # ========================================================

    print("\n========== DATA CLEANING ==========\n")

    required_columns = {
        "InvoiceNo",
        "Description",
        "Quantity",
        "UnitPrice",
        "InvoiceDate"
    }

    missing_columns = required_columns - set(raw.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    df = (
        raw
        .withColumn(
            "InvoiceNo",
            F.col("InvoiceNo").cast("string")
        )
        .withColumn(
            "Description",
            F.trim(F.col("Description").cast("string"))
        )
        .withColumn(
            "Quantity",
            F.col("Quantity").cast("double")
        )
        .withColumn(
            "UnitPrice",
            F.col("UnitPrice").cast("double")
        )
        .withColumn(
            "InvoiceDate",
            F.to_timestamp("InvoiceDate")
        )
        .dropDuplicates()
        .filter(F.col("InvoiceNo").isNotNull())
        .filter(F.col("Description").isNotNull())
        .filter(F.col("Description") != "")
        .filter(F.col("Quantity") > 0)
        .filter(F.col("UnitPrice") >= 0)
        .withColumn(
            "Revenue",
            F.col("Quantity") * F.col("UnitPrice")
        )
        .withColumn(
            "YearMonth",
            F.date_format("InvoiceDate", "yyyy-MM")
        )
    )

    clean_count = df.count()

    print("Clean rows:", clean_count)
    print("Removed rows:", raw_count - clean_count)


    # ========================================================
    # 6. PRODUCT SALES ANALYSIS
    # ========================================================

    print("\n========== PRODUCT SALES ANALYSIS ==========\n")

    top_products = (
        df.groupBy("Description")
        .agg(
            F.countDistinct("InvoiceNo").alias("Transactions"),
            F.sum("Quantity").alias("Units"),
            F.sum("Revenue").alias("Revenue")
        )
        .orderBy(F.desc("Transactions"))
    )

    print("Top 10 products:")
    top_products.show(10, truncate=False)


    # ========================================================
    # 7. MONTHLY REVENUE ANALYSIS
    # ========================================================

    print("\n========== MONTHLY REVENUE ANALYSIS ==========\n")

    monthly_revenue = (
        df.groupBy("YearMonth")
        .agg(
            F.sum("Revenue").alias("Revenue")
        )
        .orderBy("YearMonth")
    )

    print("Monthly revenue:")
    monthly_revenue.show(20, truncate=False)


    # ========================================================
    # 8. TRANSACTION BASKET CREATION
    # ========================================================

    print("\n========== TRANSACTION BASKETS ==========\n")

    transactions = (
        df.groupBy("InvoiceNo")
        .agg(
            F.array_sort(
                F.array_distinct(
                    F.collect_list("Description")
                )
            ).alias("items")
        )
        .filter(F.size("items") >= 2)
        .cache()
    )

    transaction_count = transactions.count()

    print(
        "Transactions with at least 2 unique items:",
        transaction_count
    )

    if transaction_count == 0:
        raise ValueError(
            "No transactions contain at least two unique products."
        )


    # ========================================================
    # 9. FP-GROWTH ALGORITHM
    # ========================================================

    print("\n========== FP-GROWTH ALGORITHM ==========\n")

    start = time.perf_counter()

    fp = FPGrowth(
        itemsCol="items",
        minSupport=MIN_SUPPORT,
        minConfidence=MIN_CONFIDENCE
    )

    fp_model = fp.fit(transactions)

    fp_freq = (
        fp_model.freqItemsets
        .orderBy(F.desc("freq"))
    )

    fp_rules = (
        fp_model.associationRules
        .orderBy(
            F.desc("lift"),
            F.desc("confidence")
        )
    )

    fp_elapsed = time.perf_counter() - start

    print("FP-Growth frequent itemsets:")
    fp_freq.show(20, truncate=False)

    print("FP-Growth association rules:")
    fp_rules.show(20, truncate=False)

    print("FP-Growth execution time:", round(fp_elapsed, 4), "seconds")


    # ========================================================
    # 10. APRIORI AND ECLAT BASELINE
    # ========================================================

    print("\n========== APRIORI AND ECLAT ==========\n")

    sample_limit = min(transaction_count, 2000)

    # Collect a bounded sample from Spark to the driver.
    # toLocalIterator avoids explicitly converting the DataFrame
    # into an RDD and applying a Python map operation.

    sample_baskets = [
        row["items"]
        for row in (
            transactions
            .limit(sample_limit)
            .select("items")
            .toLocalIterator()
        )
    ]

    print("Sample baskets:", len(sample_baskets))


    # --------------------------------------------------------
    # APRIORI PAIR IMPLEMENTATION
    # --------------------------------------------------------

    def apriori_pairs(baskets, min_support):

        n = len(baskets)

        if n == 0:
            return {}

        pair_counts = {}

        for basket in baskets:

            unique_items = set(basket)

            for a, b in combinations(
                sorted(unique_items), 2
            ):

                pair = (a, b)

                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + 1
                )

        frequent_pairs = {
            pair: count
            for pair, count in pair_counts.items()
            if count / n >= min_support
        }

        return frequent_pairs


    # --------------------------------------------------------
    # ECLAT PAIR IMPLEMENTATION
    # --------------------------------------------------------

    def eclat_pairs(baskets, min_support):

        n = len(baskets)

        if n == 0:
            return {}

        # Vertical representation:
        # item -> set of transaction IDs

        tidsets = {}

        for tid, basket in enumerate(baskets):

            for item in set(basket):

                tidsets.setdefault(
                    item, set()
                ).add(tid)

        items = sorted(tidsets.keys())

        frequent_pairs = {}

        for i, a in enumerate(items):

            for b in items[i + 1:]:

                intersection = (
                    tidsets[a] & tidsets[b]
                )

                support_count = len(intersection)

                if support_count / n >= min_support:

                    frequent_pairs[(a, b)] = (
                        support_count
                    )

        return frequent_pairs


    # --------------------------------------------------------
    # EXECUTION AND PERFORMANCE COMPARISON
    # --------------------------------------------------------

    t0 = time.perf_counter()

    apr = apriori_pairs(
        sample_baskets,
        MIN_SUPPORT
    )

    apr_time = time.perf_counter() - t0


    t0 = time.perf_counter()

    ecl = eclat_pairs(
        sample_baskets,
        MIN_SUPPORT
    )

    ecl_time = time.perf_counter() - t0


    print("\nApriori Results:")
    print("Frequent pairs:", len(apr))
    print("Execution time:", round(apr_time, 4), "seconds")


    print("\nEclat Results:")
    print("Frequent pairs:", len(ecl))
    print("Execution time:", round(ecl_time, 4), "seconds")


    print("\nFP-Growth Results:")
    print("Execution time:", round(fp_elapsed, 4), "seconds")


    # ========================================================
    # 11. SAVE OUTPUT FILES
    # ========================================================

   # ========================================================
# 11. SAVE OUTPUT FILES (WINDOWS COMPATIBLE)
# ========================================================

print("\n========== SAVING RESULTS ==========\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_dataframe_csv(dataframe, folder_name):

    output_path = os.path.join(OUTPUT_DIR, folder_name)
    # Remove previous output if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)

    csv_file = os.path.join(
        output_path,
        "part-00000.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        # Write column headers
        writer.writerow(dataframe.columns)

        # Write Spark DataFrame rows using local Python CSV
        for row in dataframe.toLocalIterator():

            values = []

            for value in row:

                if value is None:
                    values.append("")
                     elif isinstance(value, (list, tuple, dict)):
                    values.append(json.dumps(value))

                else:
                    values.append(value)

            writer.writerow(values)

    print("Saved:", csv_file)


# Save top products
save_dataframe_csv(
    top_products,
    "top_products"
)
# Save monthly revenue
save_dataframe_csv(
    monthly_revenue,
    "monthly_revenue"
)

# Save FP-Growth frequent itemsets
save_dataframe_csv(
    fp_freq,
    "fp_growth_itemsets"
)

# Save FP-Growth association rules
save_dataframe_csv(
    fp_rules,
    "fp_growth_rules"
    )

print("\nAll CSV files saved successfully!")

  

    # ========================================================
    # 12. SAVE PERFORMANCE REPORT
    # ========================================================

    performance_file = os.path.join(
        OUTPUT_DIR,
        "performance.txt"
    )

    with open(
        performance_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("MARKET BASKET ANALYSIS PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Python version: {sys.version}\n")
        f.write(f"Spark version: {spark.version}\n")

        f.write(f"Raw rows: {raw_count}\n")
        f.write(f"Clean rows: {clean_count}\n")

        f.write(
            f"Transactions with >=2 items: {transaction_count}\n"
        )

        f.write(
            f"Sample size for Apriori/Eclat: {sample_limit}\n"
        )

        f.write(
            f"Minimum support: {MIN_SUPPORT}\n"
        )

        f.write(
            f"Minimum confidence: {MIN_CONFIDENCE}\n"
        )

        f.write(
            f"Apriori frequent pairs: {len(apr)}\n"
        )

        f.write(
            f"Eclat frequent pairs: {len(ecl)}\n"
        )

        f.write(
            f"Apriori execution time: {apr_time:.6f} seconds\n"
        )

        f.write(
            f"Eclat execution time: {ecl_time:.6f} seconds\n"
        )

        f.write(
            f"FP-Growth execution time: {fp_elapsed:.6f} seconds\n"
        )


    # ========================================================
    # 13. PROJECT COMPLETION
    # ========================================================

    print("\n============================================")
    print(" MARKET BASKET ANALYSIS COMPLETED SUCCESSFULLY")
    print("============================================")

    print("\nGenerated output folders:")

    print(os.path.join(OUTPUT_DIR, "top_products"))
    print(os.path.join(OUTPUT_DIR, "monthly_revenue"))
    print(os.path.join(OUTPUT_DIR, "fp_growth_itemsets"))
    print(os.path.join(OUTPUT_DIR, "fp_growth_rules"))
    print(performance_file)

    print("\nAll results saved successfully!")


finally:

    # ========================================================
    # 14. CLEANUP
    # ========================================================

    transactions.unpersist() if "transactions" in locals() else None

    spark.stop()

    print("\nSpark session stopped.")

