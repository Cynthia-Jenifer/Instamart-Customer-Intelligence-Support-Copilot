
"""Pandas cross-check and exploratory data analysis."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def pandas_analysis():

    # ----------------------------------
    # LOAD CLEANED DATA
    # ----------------------------------

    o = pd.read_csv(CLEAN / "orders.csv")
    cust = pd.read_csv(CLEAN / "customers.csv")
    prod = pd.read_csv(CLEAN / "products.csv")

    # Keep only completed orders
    completed = o.loc[o.status.eq("completed")].copy()

    # Make sure numeric columns are numeric
    numeric_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]

    for col in numeric_columns:
        completed[col] = pd.to_numeric(
            completed[col], errors="coerce"
        )

    # ----------------------------------
    # PART A: PIVOT TABLE CROSS-CHECK
    # ----------------------------------

    # Add city and category to completed orders
    analysis = (
        completed
        .merge(cust[["customer_id", "city"]], on="customer_id")
        .merge(prod[["product_id", "category"]], on="product_id")
    )

    pivot = pd.pivot_table(
        analysis,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Grand Total"
    )

    print("\nQ2 PIVOT TABLE: REVENUE BY CITY AND CATEGORY")
    print(pivot.round(2))

    sql_total = completed.order_amount.sum()
    pivot_total = pivot.loc["Grand Total", "Grand Total"]

    print("\nSQL total revenue:", round(sql_total, 2))
    print("Pivot table grand total:", round(pivot_total, 2))

    assert round(sql_total, 2) == 2998668.80
    assert round(pivot_total, 2) == round(sql_total, 2)

    # ----------------------------------
    # GROUPBY WITH NAMED AGGREGATION
    # ----------------------------------

    city_summary = (
        completed
        .merge(cust[["customer_id", "city"]], on="customer_id")
        .groupby("city")
        .agg(
            total_revenue=("order_amount", "sum"),
            average_order_amount=("order_amount", "mean"),
            order_count=("order_amount", "count")
        )
        .round(2)
        .sort_index()
    )

    print("\nCOMPLETED ORDER SUMMARY BY CITY")
    print(city_summary)
    print("\n" + "=" * 65)
    print("             SALES PERFORMANCE BY CITY")
    print("=" * 65)
    display_city = city_summary.rename(columns={
    "total_revenue": "Total Sales (INR)",
    "average_order_amount": "Average Order Value (INR)",
    "order_count": "Number of Orders"
    })
    print(display_city.to_string(
    formatters={
        "Total Sales (INR)": lambda x: f"₹{x:,.2f}",
        "Average Order Value (INR)": lambda x: f"₹{x:,.2f}",
        "Number of Orders": lambda x: f"{x:,.0f}"
    }
))
    #Chart 1 — Which city generates the most revenue?
    plt.figure(figsize=(10, 5))
    city_summary["total_revenue"].sort_values().plot(
    kind="barh"
    )
    plt.title("Total Sales by City")
    plt.xlabel("Total Sales (₹)")
    plt.ylabel("City")
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(REPORTS / "city_revenue.png", dpi=150)
    plt.show()   
   
    # Check two reference cities
    assert round(city_summary.loc["Bengaluru", "total_revenue"], 2) == 523643.40
    assert round(city_summary.loc["Bengaluru", "average_order_amount"], 2) == 276.47
    assert city_summary.loc["Bengaluru", "order_count"] == 1894

    assert round(city_summary.loc["Mumbai", "total_revenue"], 2) == 479058.50
    assert round(city_summary.loc["Mumbai", "average_order_amount"], 2) == 297.00
    assert city_summary.loc["Mumbai", "order_count"] == 1613

    # ----------------------------------
    # PART B: DESCRIPTIVE STATISTICS
    # ----------------------------------

    amounts = completed.order_amount.dropna()

    mean_value = amounts.mean()
    median_value = amounts.median()
    skew_value = amounts.skew()

    q1 = amounts.quantile(0.25)
    q2 = amounts.quantile(0.50)
    q3 = amounts.quantile(0.75)

    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    lower_outliers = (amounts < lower_fence).sum()
    upper_outliers = (amounts > upper_fence).sum()

    print("\nCOMPLETED ORDER AMOUNT STATISTICS")
    print("Mean:", round(mean_value, 2))
    print("Median:", round(median_value, 2))
    print("Skewness:", round(skew_value, 2))
    print("Q1:", round(q1, 2))
    print("Q2 (Median):", round(q2, 2))
    print("Q3:", round(q3, 2))
    print("IQR:", round(iqr, 2))
    print("Lower fence:", round(lower_fence, 2))
    print("Upper fence:", round(upper_fence, 2))
    print("Outliers below lower fence:", lower_outliers)
    print("Outliers above upper fence:", upper_outliers)

 #Chart 2 — What do order values look like?
    plt.figure(figsize=(10, 5))

    plt.hist(
        amounts,
        bins=30,
        edgecolor="black"
    )

    plt.axvline(
        amounts.median(),
        linestyle="--",
        label="Median: ₹135"
    )

    plt.axvline(
        amounts.mean(),
        linestyle="--",
        label="Average: ₹269.91"
    )

    plt.title("Distribution of Completed Order Values")
    plt.xlabel("Order Value (₹)")
    plt.ylabel("Number of Orders")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "order_value_distribution.png", dpi=150)
    plt.show()


    assert round(mean_value, 2) == 269.91
    assert round(median_value, 2) == 135.00
    assert round(skew_value, 2) == 4.61
    assert round(q1, 2) == 64.80
    assert round(q3, 2) == 297.00
    assert round(upper_fence, 2) == 645.30
    assert lower_outliers == 0
    assert upper_outliers == 897

    #unusable orders
    outlier_percentage = len(amounts[amounts > upper_fence]) / len(amounts) * 100

    print("\nUNUSUALLY HIGH-VALUE ORDERS")
    print("-" * 50)
    print(f"Orders above ₹{upper_fence:,.2f}: {upper_outliers:,}")
    print(f"Percentage of completed orders: {outlier_percentage:.2f}%")
    print("Orders below the lower limit:", lower_outliers)

    print(
        "\nBusiness interpretation: Some orders are much more expensive "
        "than a typical order. These may be bulk purchases or special orders. "
        "They should be investigated, not automatically deleted."
    )

    # ----------------------------------
    # CORRELATION MATRIX
    # ----------------------------------

    correlation_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]

    correlation_matrix = completed[correlation_columns].corr()

    print("\nCORRELATION MATRIX")
    print(correlation_matrix.round(3))

    # Find the pair with the largest absolute correlation.
    # Ignore the diagonal and duplicate pairs.
    upper_triangle = correlation_matrix.where(
        pd.DataFrame(
            True,
            index=correlation_matrix.index,
            columns=correlation_matrix.columns
        ).where(
            pd.DataFrame(
                [
                    [i < j for j in range(len(correlation_columns))]
                    for i in range(len(correlation_columns))
                ],
                index=correlation_matrix.index,
                columns=correlation_matrix.columns
            )
        )
    )

    strongest_pair = (
        upper_triangle.stack()
        .abs()
        .idxmax()
    )

    strongest_value = correlation_matrix.loc[
        strongest_pair[0], strongest_pair[1]
    ]

    print("\nStrongest correlation pair:", strongest_pair)
    print("Correlation value:", round(strongest_value, 3))

    assert strongest_pair == ("order_amount", "discount_pct")
    assert round(strongest_value, 3) == -0.095

    #correlations
    outlier_percentage = len(amounts[amounts > upper_fence]) / len(amounts) * 100

    print("\nUNUSUALLY HIGH-VALUE ORDERS")
    print("-" * 50)
    print(f"Orders above ₹{upper_fence:,.2f}: {upper_outliers:,}")
    print(f"Percentage of completed orders: {outlier_percentage:.2f}%")
    print("Orders below the lower limit:", lower_outliers)

    print(
        "\nBusiness interpretation: Some orders are much more expensive "
        "than a typical order. These may be bulk purchases or special orders. "
        "They should be investigated, not automatically deleted."
    )
    print("\nALL PANDAS CHECKPOINTS PASSED")


if __name__ == "__main__":
    pandas_analysis()
