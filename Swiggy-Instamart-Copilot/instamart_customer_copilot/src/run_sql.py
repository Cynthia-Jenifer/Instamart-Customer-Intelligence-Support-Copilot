
"""Run the seven SQL analysis queries and check assignment results."""

from pathlib import Path
import sqlite3
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"


def run_queries():

    # Read all SQL queries from the file
    sql_text = SQL_FILE.read_text(encoding="utf-8")

    # Split the file wherever a Q1, Q2, ... comment appears
    query_parts = re.split(r"(?m)^-- Q[1-7].*$", sql_text)

    # Remove empty sections
    queries = [part.strip() for part in query_parts if part.strip()]

    assert len(queries) == 7, "Expected exactly 7 SQL queries"

    results = {}

    # Connect to SQLite and execute each query
    with sqlite3.connect(DB) as con:

        for i, query in enumerate(queries, start=1):

            df = pd.read_sql_query(query, con)
            results[f"Q{i}"] = df

            print(f"\nQ{i} RESULTS")
            print(df.head())
            print("Number of rows:", len(df))

    # ----------------------------------
    # CHECK REFERENCE RESULTS
    # ----------------------------------

    q1 = results["Q1"]
    q2 = results["Q2"]
    q3 = results["Q3"]
    q4 = results["Q4"]
    q5 = results["Q5"]
    q6 = results["Q6"]
    q7 = results["Q7"]

    assert len(q1) == 30

    assert len(q2) == 30
    assert round(q2.revenue.sum(), 2) == 2998668.80

    largest_cell = q2.loc[q2.revenue.idxmax()]
    assert largest_cell.city == "Bengaluru"
    assert largest_cell.category == "Electronics"
    assert round(largest_cell.revenue, 2) == 233909.45

    assert len(q3) == 11

    assert len(q4) == 25

    assert len(q5) == 550
    assert round(q5.average_spend.iloc[0], 2) == 2552.06

    assert len(q6.loc[q6.spend_rank != q6.row_num]) == 8

    expected_top_customers = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10)
    }

    top_customers = q6.loc[q6.spend_rank == 1]

    for city, (customer_id, spend) in expected_top_customers.items():

        row = top_customers.loc[top_customers.city == city].iloc[0]

        assert row.customer_id == customer_id
        assert round(row.total_spend, 2) == spend

    assert len(q7) == 183
    assert round(q7.cumulative_revenue.iloc[-1], 2) == 2998668.80

    print("\nALL SQL CHECKPOINTS PASSED")


if __name__ == "__main__":
    run_queries()
