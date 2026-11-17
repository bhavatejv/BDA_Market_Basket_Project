# Market Basket Analysis for Retail Sales Optimization — CSE557 BDA

## Project basis
This project follows the uploaded 2024 ICETITE research paper *Market Basket Analysis for Retail Sales Optimization* and the MSRIT CSE557 "Dataset Analysis for Real-Time Scenarios" requirements.

The paper proposes Apriori, FP-Growth and Eclat for discovering frequent itemsets and association rules, using support, confidence and lift, followed by product-placement optimization. The CSE557 template requires a real-world scenario, a recent 2024–2026 paper, dataset preparation, Spark/PySpark implementation, visualizations, performance evaluation and comparison.

## Dataset
**Primary real dataset:** UCI Online Retail  
URL: https://archive.ics.uci.edu/dataset/352/online+retail  
UCI reports 541,909 instances and six displayed feature roles; the commonly downloaded workbook contains the retail transaction fields used by this project.

**Testing dataset included in this folder:** `data/simulated_retail_transactions.csv`  
This is **simulated** and is clearly labelled as such. It is included so the workflow can be tested without downloading the external dataset.

## Run
1. Install Python 3.10+ and Java 8/11/17.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Test with the included simulated data:
   `spark-submit market_basket_spark_project.py`
4. For the UCI dataset, download `Online Retail.xlsx`, convert it to CSV with equivalent column names, and run:
   `INPUT_PATH=data/online_retail.csv spark-submit market_basket_spark_project.py`

## Outputs
The Spark script creates:
- `top_products`
- `monthly_revenue`
- `fp_growth_itemsets`
- `fp_growth_rules`
- `performance.txt`

The included `results/` folder contains demonstrative results and charts generated from the simulated dataset. These are **not** claimed as results from the UCI dataset.

## Research-paper alignment
- Data preprocessing and transaction cleaning
- Apriori / FP-Growth / Eclat comparison
- Support, confidence and lift
- Frequent itemset mining
- Product recommendation logic
- Product-placement/layout insight
- Execution-time comparison
- Spark-based scalable processing

## Important academic note
Do not present the simulated-data charts as UCI results. When submitting with the real UCI dataset, rerun the Spark pipeline and replace the demonstrative results with the generated outputs.
