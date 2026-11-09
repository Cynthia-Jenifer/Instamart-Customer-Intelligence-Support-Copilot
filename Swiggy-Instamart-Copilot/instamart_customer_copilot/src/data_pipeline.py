"""Part A: audit, clean, validate and load data into SQLite."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"; CLEAN=ROOT/"data"/"clean"; DB=ROOT/"data"/"instamart.db"

def audit():
    for name in ["customers", "products", "orders", "tickets"]:

        df = pd.read_csv(
            RAW / f"{name}.csv",
            keep_default_na=False
        )

        print(f"\n{name}: shape={df.shape}")

        print("\nData Types:")
        print(df.dtypes)

        # Count NaN values and blank strings
        missing_count = (
            df.isna().sum()
            + df.astype(str).apply(
                lambda col: col.str.strip().eq("")
            ).sum()
        )

        missing_pct = (
            missing_count / len(df) * 100
        ).round(2)

        missing_report = pd.DataFrame({
            "missing_count": missing_count,
            "missing_pct": missing_pct
        })

        print("\nMissing Values:")
        print(missing_report)

    # Orders audit
    orders = pd.read_csv(
        RAW / "orders.csv",
        keep_default_na=False
    )

    cust = pd.read_csv(
        RAW / "customers.csv",
        keep_default_na=False
    )

    print("\nOrders audit:")

    orders_audit = {
        "rows": len(orders),
        "exact_duplicates": int(orders.duplicated().sum()),
        "pending": int(orders.status.eq("pending").sum()),
        "distinct_order_id": int(orders.order_id.nunique()),
        "bad_amount": int(
            pd.to_numeric(
                orders.order_amount,
                errors="coerce"
            ).isna().sum()
        ),
        "negative_delivery": int(
            pd.to_numeric(
                orders.delivery_minutes,
                errors="coerce"
            ).lt(0).sum()
        ),
        "blank_delivery": int(
            orders.delivery_minutes.astype(str)
            .str.strip().eq("").sum()
        )
    }

    print(orders_audit)

    # Customers audit
    print("\nCustomer audit:")

    customers_audit = {
        "blank_age": int(
            cust.age.astype(str).str.strip().eq("").sum()
        ),
        "age_outside_10_100": int(
            pd.to_numeric(cust.age, errors="coerce")
            .pipe(lambda x: ((x < 10) | (x > 100)).sum())
        ),
        "raw_city_spellings": int(cust.city.nunique()),
        "raw_gender_spellings": int(
            cust.loc[
                cust.gender.astype(str).str.strip().ne(""),
                "gender"
            ].nunique()
        ),
        "blank_gender": int(
            cust.gender.astype(str).str.strip().eq("").sum()
        )
    }

    print(customers_audit)
    
    # Combined audit figures in a table
    audit_table = pd.DataFrame([
        {"table": "orders.csv", "metric": k, "count": v}
        for k, v in orders_audit.items()
    ] + [
        {"table": "customers.csv", "metric": k, "count": v}
        for k, v in customers_audit.items()
    ])

    print("\nFINAL RAW DATA AUDIT TABLE:")
    print(audit_table.to_string(index=False))
    
def clean_data():
    CLEAN.mkdir(parents=True,exist_ok=True)
    cust=pd.read_csv(RAW/"customers.csv",keep_default_na=False)
    prod=pd.read_csv(RAW/"products.csv")
    orders=pd.read_csv(RAW/"orders.csv",keep_default_na=False)
    # (a) Standardise city names
    cust["city"]=cust.city.astype(str).str.strip().str.title()
    # (b) Standardise gender values
    cust["gender"]=cust.gender.astype(str).str.strip().str.lower().map({"male":"male","m":"male","female":"female","f":"female"}).fillna("unknown")
    # (c) Convert age into numeric values
    cust["age"]=pd.to_numeric(cust.age,errors="coerce")
    # Treat ages outside 10–100 as missing
    cust.loc[~cust.age.between(10,100),"age"]=np.nan
    # Calculate each city's median age
    cust["age"]=cust.age.fillna(cust.groupby("city").age.transform("median")).round().astype(int)
    # (d) Convert signup_date to datetime
    cust["signup_date"]=pd.to_datetime(cust.signup_date).dt.strftime("%Y-%m-%d")
    orders["order_ts"]=pd.to_datetime(orders.order_ts)
    orders=orders.drop_duplicates()
    # Latest timestamp wins when the same order appears as pending and final.
    orders=orders.sort_values(["order_id","order_ts"]).drop_duplicates("order_id",keep="last").copy()
    orders["quantity"]=pd.to_numeric(orders.quantity,errors="coerce")
    orders["discount_pct"]=pd.to_numeric(orders.discount_pct,errors="coerce").fillna(0)
    orders["order_amount"]=pd.to_numeric(orders.order_amount,errors="coerce")
    orders["delivery_minutes"]=pd.to_numeric(orders.delivery_minutes,errors="coerce")
    orders.loc[orders.delivery_minutes<0,"delivery_minutes"]=np.nan
    price=prod.set_index("product_id").price_inr
    missing=orders.order_amount.isna()
    orders.loc[missing,"order_amount"]=(orders.loc[missing,"product_id"].map(price)*orders.loc[missing,"quantity"]*(1-orders.loc[missing,"discount_pct"]/100)).round(2)
    orders["category"]=orders.product_id.map(prod.set_index("product_id").category)
    completed=orders.status.eq("completed")
    med=orders.loc[completed].groupby("category").delivery_minutes.transform("median")
    # Category medians mapped to all completed rows, including missing values.
    cat_median=orders.loc[completed].groupby("category").delivery_minutes.median()
    fillmask=completed & orders.delivery_minutes.isna()
    orders.loc[fillmask,"delivery_minutes"]=orders.loc[fillmask,"category"].map(cat_median)
    orders["delivery_minutes"]=orders.delivery_minutes.round().astype("Int64")
    orders=orders.drop(columns=["category"])
    orders["order_ts"]=orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    for name,df in [("customers",cust),("products",prod),("orders",orders)]:
        df.to_csv(CLEAN/f"{name}.csv",index=False)
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:
        for name,df in [("customers",cust),("products",prod),("orders",orders)]:
            df.to_sql(name,con,if_exists="replace",index=False)
    print("\nClean CSV files and SQLite tables saved successfully!")

    return {"customers":cust,"products":prod,"orders":orders}
def validate():
    # Load cleaned CSV files
    c=pd.read_csv(CLEAN/"customers.csv"); o=pd.read_csv(CLEAN/"orders.csv",keep_default_na=False)
    # CUSTOMER CHECKPOINTS
    assert len(c) == 1200
    assert c.city.nunique()==6
    assert c.city.value_counts().to_dict() == {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184
    }

    assert c.gender.value_counts().to_dict() == {
        "female": 610,
        "male": 560,
        "unknown": 30
    }
    assert set(c.gender.unique())<={"male","female","unknown"}
    assert c.age.notna().all() and c.age.between(18,60).all()
    # ORDER CHECKPOINTS
    assert len(o) == 12309
 
    assert o.order_id.is_unique
    assert o.status.value_counts().to_dict() == {
        "completed": 11110,
        "cancelled": 1199
    }
    assert not o.status.eq("pending").any()
    assert pd.to_numeric(o.order_amount,errors="coerce").notna().all()
    d=pd.to_numeric(o.loc[o.status.eq("completed"),"delivery_minutes"],errors="coerce")
    assert d.notna().all()
    assert pd.to_numeric(o.loc[o.status.eq("cancelled"),"delivery_minutes"],errors="coerce").isna().all()

     # Display customer validation results
    print("\nCUSTOMER VALIDATION RESULTS")

    print("\nCity distribution:")
    print(c.city.value_counts())

    print("\nGender distribution:")
    print(c.gender.value_counts())
    print("\nMissing ages:",  c.age.isna().sum())

    print("Minimum age:", c.age.min())

    print("Maximum age:", c.age.max())
    print("\nORDER VALIDATION RESULTS")
    print("Total orders:", len(o))
    print("\nStatus distribution:")
    print(o.status.value_counts()) 
    print("Missing order amounts:",pd.to_numeric(o.order_amount, errors="coerce").isna().sum())
    completed = o.status.eq("completed")
    cancelled = o.status.eq("cancelled")
    print("Missing completed delivery times:",pd.to_numeric(o.loc[completed, "delivery_minutes"],errors="coerce").isna().sum())
    print("Missing cancelled delivery times:",pd.to_numeric(o.loc[cancelled, "delivery_minutes"],errors="coerce").isna().sum())
    print("Completed revenue:",pd.to_numeric(o.loc[completed, "order_amount"],errors="coerce").sum())
    with sqlite3.connect(DB) as con:
        names={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        assert {"customers","products","orders"}<=names
        
    print("ALL CHECKPOINTS PASSED")
if __name__=="__main__":
    audit(); clean_data(); validate()
