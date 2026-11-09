"""Part B: RFM segmentation and a leakage-aware lapse model."""
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/"data/clean"; OUT=ROOT/"outputs"
SNAPSHOT=pd.Timestamp("2025-10-01")
def load():
    c=pd.read_csv(CLEAN/"customers.csv"); o=pd.read_csv(CLEAN/"orders.csv")
    o["order_ts"]=pd.to_datetime(o.order_ts)
    return c,o
def rfm_and_segments(k=4):
    c,o=load(); done=o[o.status.eq("completed")].copy()
    rfm=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(SNAPSHOT-rfm.last_order).dt.days; rfm=rfm.drop(columns="last_order")
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[["recency","frequency","monetary"]])
    diagnostics=[]
    from sklearn.metrics import silhouette_score
    for n in range(2,9):
        km=KMeans(n_clusters=n,n_init=10,random_state=42).fit(X)
        diagnostics.append({"k":n,"inertia":km.inertia_,"silhouette":silhouette_score(X,km.labels_)})
    # Fit final K-Means model
    km=KMeans(n_clusters=k,n_init=10,random_state=42); rfm["segment_id"]=km.fit_predict(X)
    # 1. Profile each segment
    profile=rfm.groupby("segment_id").agg(size=("customer_id","size"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).reset_index()
    order=profile.sort_values("monetary",ascending=False).segment_id.tolist()
    names=["Premium" if i==order[0] else ("Frequent" if profile.set_index("segment_id").loc[i,"frequency"]>=profile.frequency.median() else "At Risk") for i in profile.segment_id]
    name_map=dict(zip(profile.segment_id,names))
    rfm["segment_name"]=rfm.segment_id.map(name_map); rfm["is_premium"]=rfm.segment_id.eq(order[0])
    OUT.mkdir(parents=True,exist_ok=True)
    rfm[["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]].to_csv(OUT/"customer_segments.csv",index=False)
  
    print("\nFirst 5 Customer Segments:")
    print(rfm[[
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]].head().to_string(index=False))
    #5. PCA on scaled RFM matrix
    pca=PCA(n_components=2).fit(X)
    # Diagnostic output for RFM
    diagnostics = pd.DataFrame(diagnostics)
    print("\nK-Means Diagnostics:")
    print(diagnostics.to_string(index=False))
    plt.figure(figsize=(8, 5))

    plt.plot(
        diagnostics["k"],
        diagnostics["inertia"],
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Method")
    plt.xticks(range(2, 9))
    plt.grid(True)

    plt.savefig(OUT / "kmeans_elbow.png", dpi=150)
    plt.show()
    return rfm,pd.DataFrame(diagnostics),profile,pca.explained_variance_ratio_
def build_lapse_model():
    c,o=load(); cutoff=pd.Timestamp("2025-09-01"); end=pd.Timestamp("2025-10-01")
    # Only orders before cutoff
    hist=o[(o.order_ts<cutoff)].copy(); 
    # Completed historical orders
    done=hist[hist.status.eq("completed")]
    #Build the customer cohort and historical features
    cohort=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum"),
      avg_discount=("discount_pct","mean"),avg_delivery=("delivery_minutes","mean")).reset_index()
    cohort["recency"]=(cutoff-cohort.last_order).dt.days; cohort=cohort.drop(columns="last_order")
    # Calculate cancellation rate
    all_final=hist[hist.status.isin(["completed","cancelled"])].groupby("customer_id").status.agg(total="count",cancelled=lambda x:(x=="cancelled").sum()).reset_index()
    cohort=cohort.merge(all_final,on="customer_id",how="left"); cohort["cancellation_rate"]=cohort.cancelled/cohort.total
    # Calculate most frequent product category
    cats=done.merge(pd.read_csv(CLEAN/"products.csv")[["product_id","category"]],on="product_id").groupby(["customer_id","category"]).size().rename("n").reset_index()
    fav=cats.sort_values(["customer_id","n","category"],ascending=[True,False,True]).drop_duplicates("customer_id").rename(columns={"category":"favourite_category"})[["customer_id","favourite_category"]]
    cohort=cohort.merge(fav,on="customer_id",how="left").merge(c,on="customer_id",how="left")
    #Add customer information and tenure
    cohort["signup_date"]=pd.to_datetime(cohort.signup_date); cohort["tenure_days"]=(cutoff-cohort.signup_date).dt.days
    #Create lapse label using September orders
    post=o[(o.order_ts>=cutoff)&(o.order_ts<end)&o.status.eq("completed")].groupby("customer_id").size()
    cohort["lapsed"]=(~cohort.customer_id.isin(post.index)).astype(int)
    #Remove unnecessary columns and define features
    cohort=cohort.drop(columns=["signup_date","total","cancelled"])
    features=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","favourite_category","city","age","gender","tenure_days"]
    X=cohort[features].copy(); y=cohort.lapsed
    numeric=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","age","tenure_days"]
    categorical=["favourite_category","city","gender"]
    # 10. Check missing values in original feature columns
    print("\n========== FEATURE MISSING VALUE CHECK ==========")

    print(X.isna().sum())

    print(
        "\nTotal missing values before preprocessing:",
        X.isna().sum().sum()
    )
    X[numeric]=X[numeric].fillna(X[numeric].median()); X[categorical]=X[categorical].fillna("unknown")
    print("\n========== LAPSE PROBLEM VALIDATION ==========")

    print("Total cohort customers:", len(cohort))

    print("\nLapsed customer count:")
    print(y.value_counts())

    print("\nLapsed percentage:")
    print(round(y.mean() * 100, 2), "%")

    print("\nNot lapsed percentage:")
    print(round((1 - y.mean()) * 100, 2), "%")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    baseline_predictions = np.zeros(len(y_test), dtype=int)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        zero_division=0
    )

    print("\n========== BASELINE MODEL ==========")
    print("Baseline accuracy:", round(baseline_accuracy, 3))
    print("Baseline lapsed F1:", round(baseline_f1, 3))
    prep=ColumnTransformer([("num",StandardScaler(),numeric),("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),categorical)])
    models={}
    for name,est,grid in [
      ("logistic",LogisticRegression(max_iter=1000),{"model__C":[.01,.1,1,10,100]}),
      ("forest",RandomForestClassifier(n_estimators=100,random_state=42),{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]})]:
        pipe=Pipeline([("prep",prep),("model",est)])
        search=GridSearchCV(pipe,grid,cv=5,scoring="f1",n_jobs=-1);
        # Tune using training data only
        search.fit(X_train,y_train)
        # Evaluate the tuned model on the untouched test set
        pred=search.predict(X_test)
        models[name]={"best_params":search.best_params_,"cv_f1":search.best_score_,"precision":precision_score(y_test,pred,zero_division=0),
          "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"confusion_matrix":confusion_matrix(y_test,pred).tolist(),"search":search}
     #  Report tuning results and test-set evaluation
    print("\n========== MODEL TUNING RESULTS ==========")

    for name, result in models.items():

        print(f"\nModel: {name.upper()}")

        print("Best parameters:", result["best_params"])

        print(
            "Best cross-validation F1:",
            round(result["cv_f1"], 3)
        )

        print("Test confusion matrix:")
        print(np.array(result["confusion_matrix"]))

        print("Test precision:", round(result["precision"], 3))
        print("Test recall:", round(result["recall"], 3))
        print("Test F1:", round(result["f1"], 3))

    # 19. Select the model with the higher test F1
    selected_name = max(
        models,
        key=lambda name: models[name]["f1"]
    )

    selected = models[selected_name]["search"]

    print("\n========== SELECTED MODEL ==========")
    print("Selected model:", selected_name.upper())
    print(
        "Selected test F1:",
        round(models[selected_name]["f1"], 3)
    )

    if models[selected_name]["f1"] < 0.55:
        print(
            "WARNING: Selected model has not reached "
            "the required test F1 of 0.55."
        )
    else:
        print("Required test F1 of 0.55 achieved.")

    print("\n========== MODEL SELECTION JUSTIFICATION ==========")

    if selected_name == "forest":
        print(
            "Random Forest was selected because it achieved "
            "the higher test F1 score for identifying lapsed "
            "customers. Missing a lapsing customer can mean "
            "losing future revenue and a customer relationship. "
            "However, wrongly targeting a loyal customer with "
            "a retention offer can waste marketing costs and "
            "reduce the value of promotional campaigns. "
            "The selected model provides a better balance of "
            "precision and recall based on the test results."
        )
    else:
        print(
            "Logistic Regression was selected because it "
            "achieved the higher test F1 score for identifying "
            "lapsed customers. Missing a lapsing customer can "
            "mean losing future revenue and a customer relationship. "
            "However, wrongly targeting a loyal customer with "
            "a retention offer can waste marketing costs and "
            "reduce the value of promotional campaigns. "
            "The selected model provides a better balance of "
            "precision and recall based on the test results."
        )
    
    # 20. CLASS-BALANCE EXPERIMENT
    print("\n========== CLASS-BALANCE EXPERIMENT ==========")

    # Get the best Logistic Regression pipeline from GridSearchCV
    best_logistic = models["logistic"]["search"].best_estimator_

    # Count the two classes in the original training set
    class_counts = y_train.value_counts()

    minority_count = class_counts.min()
    majority_count = class_counts.max()

    print("\nOriginal training class counts:")
    print(class_counts)

    # Extract only lapsed customers from training data
    X_lapsed = X_train[y_train == 1]

    # Calculate how many additional lapsed rows are needed
    rows_to_add = majority_count - minority_count

    # Randomly duplicate lapsed training rows with replacement
    X_lapsed_extra = resample(
        X_lapsed,
        replace=True,
        n_samples=rows_to_add,
        random_state=42
    )

    # Keep all original training rows and add duplicated rows
    X_balanced = pd.concat(
        [X_train, X_lapsed_extra],
        ignore_index=True
    )

    # Create labels for the additional lapsed customers
    y_lapsed_extra = pd.Series(
        np.ones(rows_to_add, dtype=int)
    )

    y_balanced = pd.concat(
        [
            y_train.reset_index(drop=True),
            y_lapsed_extra
        ],
        ignore_index=True
    )

    # Shuffle the balanced training dataset
    shuffle_indices = np.random.RandomState(42).permutation(
        len(X_balanced)
    )

    X_balanced = X_balanced.iloc[shuffle_indices]
    y_balanced = y_balanced.iloc[shuffle_indices]

    print("\nBalanced training class counts:")
    print(y_balanced.value_counts())

    print("\nBalanced training rows:", len(X_balanced))

    # Refit a fresh copy of the best Logistic Regression pipeline
    balanced_logistic = clone(best_logistic)

    balanced_logistic.fit(X_balanced, y_balanced)

    # Evaluate on the ORIGINAL unchanged test set
    balanced_predictions = balanced_logistic.predict(X_test)

    balanced_precision = precision_score(
        y_test,
        balanced_predictions,
        zero_division=0
    )

    balanced_recall = recall_score(
        y_test,
        balanced_predictions,
        zero_division=0
    )

    balanced_f1 = f1_score(
        y_test,
        balanced_predictions,
        zero_division=0
    )

    # Get the original unbalanced Logistic Regression results
    original_precision = models["logistic"]["precision"]
    original_recall = models["logistic"]["recall"]
    original_f1 = models["logistic"]["f1"]

    # Compare the results
    comparison = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1"],
        "Unbalanced": [
            original_precision,
            original_recall,
            original_f1
        ],
        "Balanced": [
            balanced_precision,
            balanced_recall,
            balanced_f1
        ]
    })

    comparison["Change"] = (
        comparison["Balanced"] - comparison["Unbalanced"]
    )

    print("\n========== BALANCED VS UNBALANCED RESULTS ==========")
    print(comparison.round(3).to_string(index=False))

    # Identify the metric with the largest absolute change
    largest_change = comparison.loc[
        comparison["Change"].abs().idxmax()
    ]

    print(
        "\nMetric that moved most:",
        largest_change["Metric"]
    )

    print(
        "Change:",
        round(largest_change["Change"], 3)
    )

    print("\n========== BUSINESS INTERPRETATION ==========")

    print(
        "Oversampling gives the model more examples of lapsed "
        "customers, helping it identify more customers at risk. "
        "Higher recall means the retention team can reach more "
        "customers before they leave. However, some loyal "
        "customers may also be incorrectly targeted, increasing "
        "marketing costs. Precision indicates how many customers "
        "identified as at risk are actually lapsed."
    )


    # FEATURE INTERPRETATION

    print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")

    best_forest = models["forest"]["search"].best_estimator_

    # Get feature names after preprocessing
    forest_feature_names = (
        best_forest.named_steps["prep"].get_feature_names_out()
    )

    forest_importances = best_forest.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": forest_feature_names,
        "importance": forest_importances
    })

    top_5_features = importance_df.sort_values(
        "importance", ascending=False
    ).head(5)

    print("\nTop 5 important features:")
    print(top_5_features.to_string(index=False))
    print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")

    best_logistic = models["logistic"]["search"].best_estimator_

    logistic_feature_names = (
        best_logistic.named_steps["prep"].get_feature_names_out()
    )

    logistic_coefficients = best_logistic.named_steps["model"].coef_[0]

    coefficient_df = pd.DataFrame({
        "feature": logistic_feature_names,
        "coefficient": logistic_coefficients
    })

    selected_features = [
        "num__recency",
        "num__frequency",
        "num__tenure_days"
    ]

    business_readings = {
        "num__recency":
            "A positive coefficient means customers who have not ordered recently tend to have higher lapse risk.",
        "num__frequency":
            "A negative coefficient means customers with more historical orders tend to have lower lapse risk.",
        "num__tenure_days":
            "A negative coefficient means longer-standing customers tend to have lower lapse risk."
    }

    for feature in selected_features:
        row = coefficient_df[coefficient_df["feature"] == feature]

        if not row.empty:
            coefficient = row["coefficient"].iloc[0]
            sign = "Positive" if coefficient > 0 else "Negative"

            print(f"\nFeature: {feature}")
            print(f"Coefficient: {coefficient:.4f}")
            print(f"Sign: {sign}")

            # The reading below is sign-dependent.
            if feature == "num__recency":
                reading = (
                    "Higher recency is associated with higher lapse odds."
                    if coefficient > 0 else
                    "Higher recency is associated with lower lapse odds."
                )
            elif feature == "num__frequency":
                reading = (
                    "Higher order frequency is associated with higher lapse odds."
                    if coefficient > 0 else
                    "Higher order frequency is associated with lower lapse odds."
                )
            else:
                reading = (
                    "Longer customer tenure is associated with higher lapse odds."
                    if coefficient > 0 else
                    "Longer customer tenure is associated with lower lapse odds."
                )

            print("Business reading:", reading)
    selected=models["forest"]["search"]
    #Compare training scaler mean with full-cohort mean
    fitted_prep = selected.best_estimator_.named_steps["prep"]

    fitted_numeric = fitted_prep.named_transformers_["num"]

    # Get the fitted StandardScaler
    fitted_scaler = (
        fitted_numeric.named_steps["scaler"]
        if isinstance(fitted_numeric, Pipeline)
        else fitted_numeric
)
    frequency_position = numeric.index("frequency")

    training_scaler_mean = fitted_scaler.mean_[frequency_position]

    full_cohort_mean = X["frequency"].mean()

    print("\n========== FREQUENCY MEAN COMPARISON ==========")

    print(
        "Training scaler mean:",
        round(training_scaler_mean, 2)
    )

    print(
        "Full-cohort frequency mean:",
        round(full_cohort_mean, 2)
    )

    print(
        "\nExplanation: The means differ because the scaler "
        "is fitted only on training rows, while the full-cohort "
        "mean includes both training and testing rows."
    )

    #Confirm no missing values after preprocessing
    X_train_processed = fitted_prep.transform(X_train)
    X_test_processed = fitted_prep.transform(X_test)
    # SAVE AND RELOAD SELECTED MODEL

    print("\n========== MODEL PERSISTENCE ==========")

    model_path = OUT / "lapse_model.joblib"

    joblib.dump(selected, model_path)
    print("Selected model saved to:", model_path)

    # Reload the saved model
    reloaded_model = joblib.load(model_path)

    # Compare predictions on the original test set
    original_predictions = selected.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)

    predictions_match = np.array_equal(
        original_predictions,
        reloaded_predictions
    )

    print("Do original and reloaded predictions match exactly?",
          predictions_match)

    assert predictions_match, "Reloaded model predictions do not match!"
    # 23. PREMIUM CUSTOMER RETENTION LIST

    print("\n========== PREMIUM CUSTOMER RETENTION LIST ==========")

    # Find the RFM output file containing customer_id and is_premium
    rfm_file = None

    for file in OUT.glob("*.csv"):
        try:
            temp = pd.read_csv(file)

            if {"customer_id", "is_premium"}.issubset(temp.columns):
                rfm_file = file
                break
        except Exception:
            continue

    if rfm_file is None:
        raise FileNotFoundError(
            "Could not find an RFM CSV containing customer_id and is_premium "
            "inside the outputs folder."
        )

    rfm_output = pd.read_csv(rfm_file)

    print("Using RFM file:", rfm_file.name)

    # Find Premium customers
    premium_ids = set(
        rfm_output.loc[
            rfm_output["is_premium"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            ),
            "customer_id"
        ]
    )

    # Get predicted lapse probabilities for test customers
    selected_probabilities = selected.predict_proba(X_test)[:, 1]

    # Keep customer IDs and actual labels aligned with X_test
    test_results = pd.DataFrame({
        "customer_id": cohort.loc[X_test.index, "customer_id"].values,
        "lapse_probability": selected_probabilities,
        "actually_lapsed": y_test.loc[X_test.index].values
    })

    # Keep only Premium customers
    premium_test = test_results[
        test_results["customer_id"].isin(premium_ids)
    ].copy()

    # Rank by predicted probability, highest first
    retention_list = premium_test.sort_values(
        "lapse_probability",
        ascending=False
    ).head(10)

    # Save required columns
    retention_list = retention_list[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]

    retention_path = OUT / "retention_list.csv"
    retention_list.to_csv(retention_path, index=False)

    print("\nTop Premium customers to consider for retention:")
    print(retention_list.to_string(index=False))

    truly_lapsed = int(retention_list["actually_lapsed"].sum())

    print("\nNumber of customers in retention list:",
          len(retention_list))

    print("Number who actually lapsed:", truly_lapsed)

    print("Retention list saved to:", retention_path)
    OUT.mkdir(exist_ok=True); joblib.dump(selected,OUT/"lapse_model.joblib")
    assert np.array_equal(selected.predict(X_test),joblib.load(OUT/"lapse_model.joblib").predict(X_test))
    # Premium list is illustrative: the flag uses whole-window RFM and is post-hoc.
    segments,_,_,_=rfm_and_segments()
    premium=set(segments.loc[segments.is_premium,"customer_id"])
    probs=selected.predict_proba(X_test)[:,list(selected.classes_).index(1)]
    ret=pd.DataFrame({"customer_id":X_test.index if X_test.index.name=="customer_id" else cohort.loc[X_test.index,"customer_id"].values,
      "lapse_probability":probs,"actually_lapsed":y_test.values})
    ret=ret[ret.customer_id.isin(premium)].sort_values("lapse_probability",ascending=False).head(10)
    ret.to_csv(OUT/"retention_list.csv",index=False)
    return {"cohort_rows":len(cohort),"lapsed":int(y.sum()),"baseline_accuracy":accuracy_score(y_test,np.zeros(len(y_test),dtype=int)),
      "models":{k:{a:b for a,b in v.items() if a!="search"} for k,v in models.items()},"retention_list":ret}

def validate_rfm(rfm):

    print("\nRFM table shape:", rfm.shape)

    print("\nRFM summary:")
    print(
        rfm[["recency", "frequency", "monetary"]]
        .agg(["median", "min", "max", "mean"])
    )

    assert len(rfm) == 1175
    assert rfm.recency.median() == 19
    assert rfm.recency.min() == 0
    assert rfm.recency.max() == 180

    assert round(rfm.frequency.mean(), 2) == 9.46
    assert rfm.frequency.max() == 33

    assert round(rfm.monetary.mean(), 2) == 2552.06
    assert rfm.monetary.max() == 14766.55

    print("\nALL RFM CHECKPOINTS PASSED")
if __name__ == "__main__":

    rfm, diagnostics, profile, pca_variance = rfm_and_segments(); 
    build_lapse_model();


    validate_rfm(rfm)

    print("\nSegment Profile:")
    print(profile.to_string(index=False))

    print("\nCluster Diagnostics:") 
    print(diagnostics.to_string(index=False))

    print("\nPCA Explained Variance:")
    print(pca_variance.round(3))
    print("Total:", round(pca_variance.sum(), 3))


    segments = pd.read_csv(OUT / "customer_segments.csv")
    retention = pd.read_csv(OUT / "retention_list.csv")

    print("Customer segments:", segments.shape)
    print("Segment columns:", segments.columns.tolist())

    print("\nRetention list:", retention.shape)
    print("Retention columns:", retention.columns.tolist())

    print("\nPremium segment monetary means:")
    print(
        segments.groupby("segment_id")["monetary"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nPremium customer count:")
    print(segments["is_premium"].sum())

    print("\nRetention list actual lapses:")
    print(retention["actually_lapsed"].sum())

    print("\nRequired files:")
    for filename in [
        "customer_segments.csv",
        "lapse_model.joblib",
        "retention_list.csv"
    ]:
        print(filename, (OUT / filename).exists())
