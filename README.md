# Customer Intelligence & Support Copilot 

## Overview

This project is an educational implementation of a fictional Instamart-style Customer Intelligence & Support Copilot. All generated records are synthetic. The project runs locally and does not require real customer data, a hosted LLM, an API key, or network calls.

The project demonstrates an end-to-end data and AI workflow:

- Load and audit CSV exports.
- Clean customer, product, and order data and persist trusted tables in SQLite.
- Perform SQL and pandas analytics.
- Build RFM customer segments and a leakage-aware lapse model.
- Redact personally identifiable information (PII), triage support tickets, retry transient failures, validate model output, and route tickets.
- Retrieve policy passages and run a bounded, database-backed refund agent with safe stopping and checkpoint recovery.

### Learning outcomes

Students learn to:

1. Read CSV exports and audit data quality.
2. Clean data and persist trusted tables in SQLite.
3. Write SQL and pandas analytics.
4. Segment customers and train a leakage-aware lapse model.
5. Redact PII and triage text with retries and output validation.
6. Retrieve policy passages and execute bounded database-backed tools.

### Repository map

- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers, and agent tool calls.
- `src/data_pipeline.py`: data audit, cleaning, SQLite loading, and validation.
- `src/customer_models.py`: RFM, K-means, PCA, and lapse model.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, and routing.
- `src/policy_rag.py`: policy chunking, TF-IDF retrieval, and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger, bounded tool loop, and checkpoint/resume.
- `sql/queries.sql`: SQL learning patterns.
- `outputs/`: generated artifacts.
- `state/`: saved agent checkpoints.
- `notebooks/Customer_Intelligence_Support_Copilot.ipynb`: notebook walkthrough.

## Reproduce

Python 3.11 or later is recommended. Run these commands from the project root. The commands below use Windows PowerShell.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate the virtual environment:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Upgrade pip:

   ```powershell
   python -m pip install --upgrade pip
   ```

4. Install project dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Generate the synthetic raw data and policy files:

   ```powershell
   python generate_data.py
   ```

6. Run Part A data cleaning, SQLite loading, and validation:

   ```powershell
   python src/data_pipeline.py
   ```
     ```powershell
   python src/run_sql.py
   ```
    ```powershell
   python src/pandas_analysis.py
   ```
   

7. Run Part B customer segmentation and lapse modelling:

   ```powershell
   python src/customer_models.py
   ```

8. Run Part C ticket triage:

   ```powershell
   python src/triage.py
   ```

9. Run Part D policy retrieval and the refund agent:

   ```powershell
   python src/policy_rag.py
   ```

   ```powershell
   python src/refund_agent.py
   ```

10. Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run its cells from top to bottom for the notebook walkthrough. The notebook uses the same project modules.

11. To inspect the final outputs, review the generated files in `outputs/`, the cleaned CSVs in `data/clean/`, the SQLite database at `data/instamart.db`, and checkpoint files in `state/`.

> If your local project uses a different script filename for a part, use the filename present in your checked-out repository. The commands above reflect the script names documented in this README.

## Part A

### Data foundation and cleaning

The pipeline standardizes city and gender values, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing order amounts from catalogue price, sets negative delivery times to missing, and imputes delivery durations only for completed orders.



Run validation with:

```powershell
python src/data_pipeline.py
```

A successful validation run prints:

```text
ALL CHECKPOINTS PASSED
```

### Reported data and SQL results

customers: shape=(1200, 5)
Missing Values:
             missing_count  missing_pct
customer_id              0         0.00
city                     0         0.00
signup_date              0         0.00
age                     61         5.08
gender                  30         2.50

products: shape=(40, 4)
Missing Values:
              missing_count  missing_pct
product_id                0          0.0
product_name              0          0.0
category                  0          0.0
price_inr                 0          0.0

orders: shape=(13084, 9)
Missing Values:
                  missing_count  missing_pct
order_id                      0         0.00
customer_id                   0         0.00
product_id                    0         0.00
quantity                      0         0.00
order_ts                      0         0.00
status                        0         0.00
discount_pct                  0         0.00
order_amount                  0         0.00
delivery_minutes           2541        19.42

tickets: shape=(164, 5)

Missing Values:
               missing_count  missing_pct
ticket_id                  0         0.00
customer_id                0         0.00
order_id                   8         4.88
text                       4         2.44
true_category              0         0.00

## 2. Audit the Raw Data
Orders audit:
{'rows': 13084, 'exact_duplicates': 119, 'pending': 656, 'distinct_order_id': 12309, 'bad_amount': 120, 'negative_delivery': 106, 'blank_delivery': 2541}

Customer audit:
{'blank_age': 61, 'age_outside_10_100': 6, 'raw_city_spellings': 24, 'raw_gender_spellings': 10, 'blank_gender': 30}

FINAL RAW DATA AUDIT TABLE:
        table               metric  count
   orders.csv                 rows  13084
   orders.csv     exact_duplicates    119
   orders.csv              pending    656
   orders.csv    distinct_order_id  12309
   orders.csv           bad_amount    120
   orders.csv    negative_delivery    106
   orders.csv       blank_delivery   2541
customers.csv            blank_age     61
customers.csv   age_outside_10_100      6
customers.csv   raw_city_spellings     24
customers.csv raw_gender_spellings     10
customers.csv         blank_gender     30

Clean CSV files and SQLite tables saved successfully!

## 3. Clean Customers
CUSTOMER VALIDATION RESULTS

City distribution:
city
Pune         227
Delhi        210
Bengaluru    197
Chennai      191
Hyderabad    191
Mumbai       184

Gender distribution:
gender
female     610
male       560
unknown     30

Missing ages: 0
Minimum age: 18
Maximum age: 60

ORDER VALIDATION RESULTS
Total orders: 12309

Status distribution:
status
completed    11110
cancelled     1199
Name: count, dtype: int64
Missing order amounts: 0
Missing completed delivery times: 0
Missing cancelled delivery times: 1199
Completed revenue: 2998668.8
ALL CHECKPOINTS PASSED

## 4. The raw orders data is cleaned using this sequence:

1. Convert `order_ts` into pandas datetime format.
2. Remove exact duplicate rows using `drop_duplicates()`.
3. Sort records by `order_id` and `order_ts`, retaining the latest record for each `order_id`. This removes earlier pending records when a later final status exists.
4. Convert `order_amount` into numeric values. Invalid amounts are treated as missing and calculated using product price × quantity × (1 − discount percentage / 100), rounded to two decimal places.
5. Convert `delivery_minutes` into numeric values and replace negative values with missing values. Fill missing delivery times for completed orders using the median delivery time of their product category, calculated with `groupby()` and `transform()`. Leave cancelled orders with missing delivery times because they were never delivered.

## 5. Data storage and validation

The cleaned customers, products, and orders datasets are saved as CSV files inside `data/clean/`. The same three datasets are loaded into `data/instamart.db` using SQLite. The database contains three tables: `customers`, `products`, and `orders`.

The `order_ts` column is stored as text in `YYYY-MM-DD HH:MM:SS` format. The validation script checks the cleaned datasets against reference checkpoints, including customer distributions, unique order IDs, order statuses, missing values, completed revenue, and SQLite table creation.

## 6. SQL Analysis
Q1 RESULTS
  customer_id  total_spend
0       C0693     13776.55
1       C0082     12317.70
2       C0743     12266.85
3       C0525     12241.30
4       C0291     12150.70
Number of rows: 30

Q2 RESULTS
        city       category    revenue
0  Bengaluru          Dairy   87075.05
1  Bengaluru    Electronics  233909.45
2  Bengaluru        Grocery  110915.65
3  Bengaluru  Personal Care   20193.85
4  Bengaluru         Snacks   71549.40
Number of rows: 30

Q3 RESULTS
  product_id         product_name  price_inr
0       P038    Bluetooth Speaker       1999
1       P033     Wireless Earbuds       1499
2       P034  Power Bank 10000mAh       1299
3       P036    Phone Charger 20W        799
4       P037           Smart Bulb        599
Number of rows: 11

Q4 RESULTS
  customer_id       city
0       C0029  Hyderabad
1       C0037       Pune
2       C0093       Pune
3       C0115  Bengaluru
4       C0120  Bengaluru
Number of rows: 25

Q5 RESULTS
  customer_id  total_spend  average_spend
0       C0693     14766.55        2552.06
1       C0291     13742.60        2552.06
2       C0488     13187.50        2552.06
3       C0082     12856.20        2552.06
4       C0743     12730.05        2552.06
Number of rows: 550

Q6 RESULTS
  customer_id       city  total_spend  spend_rank  dense_spend_rank  row_num
0       C0082  Bengaluru      12856.2           1                 1        1
1       C0525  Bengaluru      12719.7           2                 2        2
2       C0699  Bengaluru      12471.8           3                 3        3
3       C0543  Bengaluru      11195.1           4                 4        4
4       C1147  Bengaluru      10156.9           5                 5        5
Number of rows: 1175

Q7 RESULTS
          day   revenue  cumulative_revenue
0  2025-04-01  16876.50            16876.50
1  2025-04-02  19854.55            36731.05
2  2025-04-03  28048.35            64779.40
3  2025-04-04  14249.60            79029.00
4  2025-04-05  16031.50            95060.50
Number of rows: 183

ALL SQL CHECKPOINTS PASSED

## 7. Pandas Cross Check using Pivot Table

### Exploratory data analysis

The dataset contains customer, product, and order information. The analysis focuses on completed orders, revenue, customer locations, product categories, and delivery details.

| EDA measure | Result |
|---|---:|
| Average order amount | ₹269.91 |
| Median order amount | ₹135.00 |
| Order amount skewness | Approximately 4.61 |
| IQR lower fence | −₹283.50 |
| IQR upper fence | ₹645.30 |
| Upper outliers | 897 |
| Lower outliers | 0 |
| Correlation: order amount and discount percentage | Approximately −0.095 |
| Correlation: order amount and quantity | Approximately +0.087 |

The order-amount distribution is strongly right-skewed: most orders have relatively low values, while a smaller number have much higher values. High-value outliers may represent bulk or unusually expensive purchases and should be investigated rather than automatically removed. The correlations shown above are very weak and do not establish causation.

**What is this data about?**

This dataset contains Instamart customer, product and order information. The analysis focuses on completed orders, their revenue, customer locations, product categories and delivery details.

**What do its distributions and statistics look like?**

The average order amount is ₹269.91, while the median is ₹135.00. Skewness is approximately 4.61, indicating a strongly right-skewed distribution. Most orders have relatively low values, while a smaller number have much higher values.

**Are there outliers or anomalies?**

Using the 1.5 × IQR method, the upper fence is ₹645.30 and the lower fence is −₹283.50. There are 897 upper outliers and no lower outliers. These high-value orders may represent bulk or unusually expensive purchases and should be investigated rather than automatically removed.

**What relationships exist between variables?**

The strongest absolute correlation is between order amount and discount percentage, at approximately −0.095. Order amount and quantity have a correlation of approximately +0.087. Both relationships are very weak. Correlation of this size does not establish causation.


## Part B

### RFM segmentation and lapse prediction

RFM segmentation uses completed orders and a fixed snapshot date. K-means diagnostics compare K values from 2 to 8. The notebook uses K=4 as an interpretable teaching choice.

The lapse model uses orders strictly before 2025-09-01 to construct features. The label is based on September outcomes. Preprocessing is fitted only on training rows using a scikit-learn pipeline, and model selection uses `GridSearchCV`.

### 1. Build the RFM Table

RFM table shape: (1175, 7)

RFM summary:
           recency  frequency      monetary
median   19.000000   5.000000   2286.500000
min       0.000000   1.000000     12.000000
max     180.000000  33.000000  14766.550000
mean     33.558298   9.455319   2552.058553

ALL RFM CHECKPOINTS PASSED


### 2. K-means diagnostics

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Inertia generally decreases as the number of clusters increases, so the lowest inertia alone is not a reason to choose K=8. K=4 has the highest silhouette score among the tested values and provides a useful balance between separation and interpretability.

### 3. Profile,name and Project the segments
First 5 Customer Segments:
customer_id  recency  frequency  monetary  segment_id segment_name  is_premium
      C0001       14          4    512.85           0      At Risk       False
      C0002       28          7   4644.90           1      Premium        True
      C0003        7         22   3644.85           2     Frequent       False
      C0004        0          5    369.00           0      At Risk       False
      C0005      147          2    324.00           3      At Risk       False



========== FEATURE MISSING VALUE CHECK ==========
recency               0
frequency             0
monetary              0
avg_discount          0
cancellation_rate     0
avg_delivery          0
favourite_category    0
city                  0
age                   0
gender                0
tenure_days           0
dtype: int64

Total missing values before preprocessing: 0

PCA Explained Variance:
[0.634 0.218]
Total: 0.852
Customer segments: (1175, 7)
Segment columns: ['customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']

Retention list: (10, 3)
Retention columns: ['customer_id', 'lapse_probability', 'actually_lapsed']

Premium segment monetary means:
segment_id
1    7129.695652
2    3410.020525
0    1057.618234
3     685.992582
Name: monetary, dtype: float64

Premium customer count:
138

Retention list actual lapses:
1

### 4. Lapse Problem
========== LAPSE PROBLEM VALIDATION ==========
Total cohort customers: 1156

Lapsed customer count:
lapsed
0    713
1    443
Name: count, dtype: int64

Lapsed percentage:
38.32 %

Not lapsed percentage:
61.68 %

### 6. Train, Tune & Evaluate
Training rows: 924
Testing rows: 232

========== BASELINE MODEL ==========
Baseline accuracy: 0.616
Baseline lapsed F1: 0.0

========== MODEL TUNING RESULTS ==========

Model: LOGISTIC
Best parameters: {'model__C': 100}
Best cross-validation F1: 0.651
Test confusion matrix:
[[111  32]
 [ 31  58]]
Test precision: 0.644
Test recall: 0.652
Test F1: 0.648

Model: FOREST
Best parameters: {'model__max_depth': 8, 'model__min_samples_leaf': 10}
Best cross-validation F1: 0.687
Test confusion matrix:
[[110  33]
 [ 32  57]]
Test precision: 0.633
Test recall: 0.64
Test F1: 0.637

========== SELECTED MODEL ==========
Selected model: LOGISTIC
Selected test F1: 0.648
Required test F1 of 0.55 achieved.

========== MODEL SELECTION JUSTIFICATION ==========
Logistic Regression was selected because it achieved the higher test F1 score for identifying lapsed customers. Missing a lapsing customer can mean losing future revenue and a customer relationship. However, wrongly targeting a loyal customer with a retention offer can waste marketing costs and reduce the value of promotional campaigns. The selected model provides a better balance of precision and recall based on the test results.

### 7. Class Balance Experiment
========== CLASS-BALANCE EXPERIMENT ==========

Original training class counts:
lapsed
0    570
1    354
Name: count, dtype: int64

Balanced training class counts:
1    570
0    570
Name: count, dtype: int64

Balanced training rows: 1140

========== BALANCED VS UNBALANCED RESULTS ==========
   Metric  Unbalanced  Balanced  Change
Precision       0.644     0.589  -0.055
   Recall       0.652     0.854   0.202
       F1       0.648     0.697   0.049

Metric that moved most: Recall
Change: 0.202

### 8. Intepret, Persist and Use

========== BUSINESS INTERPRETATION ==========
Oversampling gives the model more examples of lapsed customers, helping it identify more customers at risk. Higher recall means the retention team can reach more customers before they leave. However, some loyal customers may also be incorrectly targeted, increasing marketing costs. Precision indicates how many customers identified as at risk are actually lapsed.

========== RANDOM FOREST FEATURE IMPORTANCE ==========

Top 5 important features:
               feature  importance
         num__monetary    0.280428
        num__frequency    0.277911
          num__recency    0.109118
     num__avg_delivery    0.059732
num__cancellation_rate    0.051834

========== LOGISTIC REGRESSION COEFFICIENTS ==========

Feature: num__recency
Coefficient: 0.0392
Sign: Positive
Business reading: Higher recency is associated with higher lapse odds.

Feature: num__frequency
Coefficient: -1.6973
Sign: Negative
Business reading: Higher order frequency is associated with lower lapse odds.

Feature: num__tenure_days
Coefficient: -0.0502
Sign: Negative
Business reading: Longer customer tenure is associated with lower lapse odds.

========== FREQUENCY MEAN COMPARISON ==========
Training scaler mean: 8.11
Full-cohort frequency mean: 8.03

Explanation: The means differ because the scaler is fitted only on training rows, while the full-cohort mean includesboth training and testing rows.

========== MODEL PERSISTENCE ==========
Selected model saved to: C:\Users\cynth\Swiggy-Instamart-Copilot\Swiggy-Instamart-Copilot\instamart_customer_copilot\outputs\lapse_model.joblib
Do original and reloaded predictions match exactly? True

========== PREMIUM CUSTOMER RETENTION LIST ==========
Using RFM file: customer_segments.csv

Top Premium customers to consider for retention:
customer_id  lapse_probability  actually_lapsed
      C0862           0.510513                0
      C0267           0.477212                0
      C1008           0.473260                0
      C0060           0.471746                0
      C0802           0.471582                0
      C0942           0.465245                0
      C0553           0.462799                0
      C0021           0.461252                0
      C0133           0.460743                0
      C0565           0.458271                1

Number of customers in retention list: 10
Number who actually lapsed: 1
Retention list saved to: C:\Users\cynth\Swiggy-Instamart-Copilot\Swiggy-Instamart-Copilot\instamart_customer_copilot\outputs\retention_list.csv


### Leakage prevention and model evaluation

- The encoder and scaler are fitted only on training data within the preprocessing pipeline; they are not fitted on test rows.
- The class-balancing resampler is applied only to the training set. The original test set remains unchanged.
- Orders dated on or after 2025-09-01 are not used to construct lapse-model features. September orders are used only to define the lapse target.
- Segment labels from Task 3 are not included as features in the lapse prediction model.

Report the final model comparison and class-balance experiment metrics from the completed run:

| Model / experiment | Precision | Recall | F1 |
|---|---:|---:|---:|
| Best model on original test set | Add measured result | Add measured result | Add measured result |
| Best logistic regression after training-only oversampling | Add measured result | Add measured result | Add measured result |

The Premium flag is derived from whole-window RFM data, which includes the September period used to define the lapse label. Therefore, the Premium retention list is illustrative and is not a leakage-free evaluation. The lapse model's predictive features themselves use only historical orders before the 2025-09-01 cutoff.

## Part C

### Ticket triage and privacy

Empty and overlong tickets are skipped before model calls. Email, phone, and order ID patterns are redacted before prompting. Retries handle simulated transient errors. JSON category values are allow-listed; malformed outputs are accepted only when exactly one allowed category can be detected using `re.findall`.

Routing rules:

- Human review receives guardrail, invalid-output, and budget cases.
- Fraud tickets go to the security team.
- Premium customers go to the premium desk.
- Other tickets go to the support queue.

### Reported ticket and redaction results

| Measure | Result |
|---|---:|
| Total tickets | 164 |
| Empty tickets skipped | 4 |
| Overlong tickets skipped | 4 |
| Long ticket token counts|734|
| Eligible tickets | 156 |
| Email redactions | 55 |
| Phone redactions | 91 |
| Order ID redactions | 132 |
| Total redactions | 278 |
| Maximum input token limit | 400 |
| Transient errors | 19 |
| Recovered error |19|
| Tickets recovered after retry |19|
| Permanent errors |0|

### Zero-shot versus few-shot results

| Prompt strategy | Classified tickets | Correct predictions | Accuracy |
|---|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.7% |
| Few-shot | 147 | 147 | 100% |

The zero-shot prompt incorrectly classified 11 fraud tickets and 10 technical tickets as billing. The few-shot prompt eliminated these classification errors.

The few-shot approach uses additional tokens. Insert the measured values from the final run below; do not leave the placeholders in the submitted README:

- Additional tokens used by few-shot: **[replace with measured `extra_tokens`]**
- Additional tokens per classified ticket: **[replace with measured `extra_tokens_per_ticket`]**

The additional few-shot token cost may be worthwhile when classification accuracy and avoiding incorrect ticket routing are more important than minimizing token usage.

## Part D

### Policy retrieval and refund tools

Policy documents are chunked with overlap, represented using `TfidfVectorizer`, and ranked using cosine similarity. The offline mock returns context-derived sentences.

Refund rules are implemented in Python tools backed by SQLite:

- Only completed orders are eligible for refund consideration.
- Completed orders delivered in more than 45 minutes qualify for a full refund.
- Orders with delivery time of 45 minutes or less receive no refund.
- Refunds above ₹1,000 are escalated instead of being automatically issued.
- Refund requests in this exercise are simulated; no real payment is made.

### TF-IDF and its limitations

Term Frequency (TF) measures how frequently a word appears in a document, while Inverse Document Frequency (IDF) gives more importance to words that are uncommon across the document collection.

A limitation of TF-IDF is that it relies on word matching rather than understanding meaning. Two sentences with different wording but the same meaning may therefore not be recognised as similar.

 ### 1. Chunk the documents
========== POLICY CHUNK AUDIT ==========
refund_policy.txt        4
delivery_policy.txt      3
escalation_policy.txt    3
payment_policy.txt       3
privacy_policy.txt       2

Total chunks: 15

### 2. Vectorise & Retrieve

========== TOP 3 RETRIEVED CHUNKS ==========

Rank: 1
Source: refund_policy.txt
Score: 0.2719
Text: order is delivered in more than 45 minutes, the customer receives a full refund of the order amount. If a completed order is delivered in 45 minutes or less, no late-delivery refund is paid. Refunds up to INR 1000 are

Rank: 2
Source: refund_policy.txt
Score: 0.2514
Text: Refund Policy. A refund can be requested only for orders whose status is completed. Orders with status cancelled were never charged, so no refund applies to them. If a completed order is delivered in more than 45 minutes, the customer

Rank: 3
Source: refund_policy.txt
Score: 0.2454
Text: late-delivery refund is paid. Refunds up to INR 1000 are approved automatically by the support system. A refundabove INR 1000 must be approved by a human support lead before money is returned. Approved refunds are credited to the original

All retrieval reference checks passed!

### 3. Augment & Answer

========== POLICY ASSISTANT ==========
Question: What happens when a completed order is delivered more than 45 minutes late?

Answer: If a completed order is delivered in 45 minutes or less, no late-delivery refund is paid.

Sources: ['refund_policy.txt', 'delivery_policy.txt']

Policy assistant test passed!

### 4. Evaluate Retrieval & Grounding
========== EVALUATION: OVERLAP 10 ==========

Answerable questions: 12
Expected document retrieved: 12/12
Expected fact found in answer: 12/12

Unanswerable questions: 2
Correct fallback answers: 2/2

========== QUESTIONS THAT FAILED ==========
No answerable questions failed.

========== POLICY CHUNK AUDIT ==========
delivery_policy.txt      3
payment_policy.txt       3
refund_policy.txt        3
escalation_policy.txt    2
privacy_policy.txt       2

Total chunks: 13

All policy chunk reference checks passed!

========== EVALUATION: OVERLAP 0 ==========

Answerable questions: 12
Expected document retrieved: 12/12
Expected fact found in answer: 10/12

Unanswerable questions: 2
Correct fallback answers: 2/2

========== QUESTIONS THAT FAILED ==========
                                                              question       expected_doc  retrieved_doc  fact_found
What delivery time makes a completed order eligible for a full refund?  refund_policy.txt           True       False
      How soon is money returned when a payment fails but is deducted? payment_policy.txt           True       False

========== COMPARISON ==========
             Metric Overlap 10 Overlap 0
 Document retrieved      12/12     12/12
Expected fact found      12/12     10/12

### 5. Tools & Tool Map

========== TOOL SPECIFICATION ==========

Tool: get_order
Description: Retrieve an order's status, amount and delivery time using its order ID.
Parameters: {'type': 'object', 'properties': {'order_id': {'type': 'string', 'description': 'The unique ID of the order.'}}, 'required': ['order_id']}

Tool: search_policy
Description: Search the policy documents and return the most relevant policy chunks.
Parameters: {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'The policy question or search keywords.'}}, 'required': ['query']}

Tool: calculate_refund
Description: Calculate the refund amount based on the order amount and delivery time.
Parameters: {'type': 'object', 'properties': {'order_amount': {'type': 'number', 'description': 'The total order amount in INR.'}, 'delivery_minutes': {'type': ['number', 'null'], 'description': 'The delivery time in minutes.'}}, 'required': ['order_amount', 'delivery_minutes']}

Tool: issue_refund
Description: Record an approved refund in the refund ledger.
Parameters: {'type': 'object', 'properties': {'order_id': {'type': 'string', 'description': 'The unique ID of the order.'}, 'refund_inr': {'type': 'number', 'description': 'The approved refund amount in INR.'}}, 'required': ['order_id', 'refund_inr']}

Tool: escalate_to_human
Description: Record a refund request that requires human support approval.
Parameters: {'type': 'object', 'properties': {'order_id': {'type': 'string', 'description': 'The unique ID of the order.'}, 'refund_inr': {'type': 'number', 'description': 'The refund amount requiring human approval in INR.'}}, 'required': ['order_id', 'refund_inr']}

========== TOOL MAP ==========
get_order -> get_order
search_policy -> search_policy
calculate_refund -> calculate_refund
issue_refund -> issue_refund
escalate_to_human -> escalate_to_human

All five tool definitions and mappings passed!

### 6. Run the ReAct Loop

Scenario CSV columns: ['scenario_id', 'customer_id', 'message', 'expected_outcome', 'expected_refund_inr']

=======================================================
Running scenario: S1
Status: REFUND
Final answer: REFUND: INR 128.25 issued.
Refund amount: 128.25
Model calls: 5
Tokens used: 897

=======================================================
Running scenario: S2
Status: REFUND
Final answer: REFUND: INR 135.0 issued.
Refund amount: 135.0
Model calls: 5
Tokens used: 897

=======================================================
Running scenario: S3
Status: ESCALATE
Final answer: ESCALATE: refund above INR 1000 sent to a human lead.
Refund amount: 1999.0
Model calls: 5
Tokens used: 884

=======================================================
Running scenario: S4
Status: DENIED
Final answer: DENIED: delivery was within 45 minutes.
Refund amount: 0.0
Model calls: 4
Tokens used: 605

=======================================================
Running scenario: S5
Status: DENIED
Final answer: DENIED: only completed orders can be refunded.
Refund amount: 0.0
Model calls: 3
Tokens used: 337

=======================================================
Running scenario: S6
Status: NOT_FOUND
Final answer: NOT_FOUND: no such order.
Refund amount: 0.0
Model calls: 2
Tokens used: 72

=======================================================
Running scenario: S7
Status: FALLBACK_MAX_ITERATIONS
Final answer: FALLBACK_MAX_ITERATIONS: Maximum iterations reached. Hand off to a human agent.
Refund amount: 0.0
Model calls: 6
Tokens used: 3000

JSONL output saved to: C:\Users\cynth\Swiggy-Instamart-Copilot\Swiggy-Instamart-Copilot\instamart_customer_copilot\outputs\agent_runs.jsonl


=====================================================================================
                 REACT AGENT SCENARIO RESULTS
=====================================================================================
Scenario    Status                         Refund (INR)     Calls      Tokens
-------------------------------------------------------------------------------------
S1          REFUND                               128.25         5         897
S2          REFUND                               135.00         5         897
S3          ESCALATE                            1999.00         5         884
S4          DENIED                                 0.00         4         605
S5          DENIED                                 0.00         3         337
S6          NOT_FOUND                              0.00         2          72
S7          FALLBACK_MAX_ITERATIONS                0.00         6        3000
=====================================================================================

### 7. Stop Safely
========== TOKEN LIMIT DEMO ==========
Status: FALLBACK_TOKEN_LIMIT
LLM calls: 3
Tokens used: 354
Final answer: FALLBACK_TOKEN_LIMIT: Token limit exceeded. Hand off to a human agent.

========== TIMEOUT DEMO ==========
Status: FALLBACK_TIMEOUT
LLM calls: 0
Tokens used: 0
Final answer: FALLBACK_TIMEOUT: Time limit reached. Hand off to a human agent.

========== CRASH AND RESUME DEMO ==========
Simulated crash: Simulated crash after checkpoint
Tools executed before crash:
['get_order', 'search_policy']
Resuming run: S1_RESUME_DEMO
Previously executed tools: ['get_order', 'search_policy']

Tools executed across both parts:
['get_order', 'search_policy', 'calculate_refund', 'issue_refund']
Final status: REFUND
Final answer: REFUND: INR 128.25 issued.
LLM calls: 5
Tokens used: 897


### Safe stopping and fallback demonstrations

The refund agent enforces three safety limits:

1. **Maximum iterations:** Stops after `max_iterations` model calls without a final answer. S7 ends with `FALLBACK_MAX_ITERATIONS`.
2. **Token limit:** Adds `usage.total_tokens` after every model call. If the total exceeds the configured limit, the agent stops before executing the requested tool. S1 with `token_limit=200` ends with `FALLBACK_TOKEN_LIMIT`.
3. **Time limit:** Checks elapsed wall-clock time using `time.monotonic()` at the beginning of each iteration. S1 with `time_limit_seconds=0` ends with `FALLBACK_TIMEOUT` before the first model call.

| Demonstration | Expected / observed status | Model calls |
|---|---|---:|
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` | 1 |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` | 0 |

All fallback responses preserve the completed tool steps and return a short human-handoff summary in `final_answer`.


### 8. Resumable agent and checkpoint recovery

The refund agent saves a checkpoint in `state/{run_id}.json` after every tool observation has been appended to the message history. Each checkpoint contains at least the run ID, scenario ID, full message list, `tokens_used`, step number, and list of tools already executed. It also saves completed steps and model-call count.

For the S1 crash-and-resume demonstration:

- A simulated crash is raised immediately after the second tool execution, after its checkpoint has been saved.
- `resume_agent(run_id)` loads the checkpoint and continues from the saved message history.
- The tools executed across the crash and resume, in order, are:

  1. `get_order`
  2. `search_policy`
  3. `calculate_refund`
  4. `issue_refund`

| Checkpoint recovery measure | Result |
|---|---|
| Tool executions before crash | 2 |
| Tool executions after resume | 2 |
| Total tool executions across both parts | 4 |
| Repeated tool executions | 0 |
| Final status | `REFUND` |
| Final refund amount | ₹128.25 |

### 9. ReAct versus plan-and-execute

Our ReAct agent makes multiple model calls because it follows an observe-and-act cycle. For example, S1 requires 5 model calls, S5 requires 3, and S6 requires 2. Each call adds to token usage, making ReAct relatively more expensive in terms of model calls and tokens. A plan-and-execute agent would ask the model once to generate the complete sequence of steps and then execute those steps using tools. It would only call the model again if an error or unexpected result required replanning. Therefore, it would generally use fewer model calls and tokens.

However, ReAct handles unexpected observations more naturally. For example, if an order returns `NOT_FOUND`, or its status is cancelled, the agent can observe the tool result and decide not to issue a refund. In plan-and-execute, a pre-generated plan may not account for every possible observation, so additional error handling and replanning would be necessary.

For this refund workflow, I would choose **ReAct** because refund decisions involve checking order status, delivery time, policy conditions, and escalation thresholds. Safety and correctness are more important than minimizing model calls. ReAct allows the agent to verify each observation before taking the next action, reducing the risk of issuing an incorrect refund.

## Limitations and Next Steps

- This is a teaching implementation using synthetic data and a deterministic, rule-based mock model, not a real hosted language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- TF-IDF retrieval is based on word matching and may miss semantically similar passages with different wording.
- The Premium flag is derived from whole-window RFM data and includes the September period used to define the lapse label. The Premium retention list is therefore illustrative rather than a leakage-free evaluation.
- The lapse model's predictive features use only orders before the 2025-09-01 cutoff.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks, robust evaluation, and human approval workflows.
- Before submission, replace all measurement prompts in Parts B, C, and D with the values from the final successful project run. Do not report estimated values as measured results.

