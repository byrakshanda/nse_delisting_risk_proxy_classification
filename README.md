# NSE Delisting-Risk Classification (Proxy Label, Random Forest)

## What this project does
Classifies NSE-listed companies into Low-Risk and High-Risk categories using 
Random Forest, based on structural features of their listing (symbol and 
company name attributes).

## Important note about the target
The source dataset contains only currently active companies (no delisted ones), 
so real delisting outcomes aren't available. I built a **proxy label** instead: 
companies with below-median listing tenure (~10.5 years) are labeled "High Risk", 
and the rest "Low Risk". This project demonstrates the Random Forest workflow — 
it is **not** a real delisting-risk predictor.

## Dataset
NSE security master data: 2,389 companies with symbol, ISIN, company name, 
listing date, and active status.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## What I did
- Checked data quality (no missing values or duplicates) and found that 
  every company was active, so a real target wasn't possible
- Built the proxy target from listing tenure using a median split (balanced classes)
- Engineered features from text fields: symbol length, company name length, 
  word count, "Limited/Ltd" flag, symbol starting with a digit
- Deliberately excluded listing-date-based features from the model to avoid leakage
- Ran EDA including a correlation heatmap (weak correlations, as expected)
- Trained a Random Forest and diagnosed overfitting (train 70.9% vs test 59.2%)
- Tuned `n_estimators` and `max_depth` by sweeping values and comparing metrics
- Analysed feature importance

## Results
| Model                                  | Accuracy | Precision | Recall | F1 Score |
|----------------------------------------|----------|-----------|--------|----------|
| Initial Random Forest (100 trees)      | 0.592    | -         | -      | -        |
| Final (100 trees, max_depth = 5)       | 0.620    | 0.61      | 0.64   | 0.62     |

**Final model:** n_estimators = 100, max_depth = 5. Limiting the depth improved 
test accuracy and reduced the train/test gap.

## Key Insight
Company name length and symbol length carried most of the (modest) predictive 
signal. The ~62% accuracy is expected, since name and symbol attributes say 
very little about how long a company has been listed.

## Next Steps
To make this a genuine delisting-risk model, merge in a real list of delisted 
or suspended companies (from NSE/BSE archives) so the target reflects actual 
outcomes, and add financial features.

## How to run it
Open `nse_delisting_risk_proxy_classification.ipynb` in Jupyter Notebook or 
Google Colab and run all cells. Requires `nse_security_master_rf.csv` in the 
same folder.
