# Dataset Instructions

This project uses the Credit Card Fraud Detection dataset from Kaggle.

## Dataset Description

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

The dataset has been anonymized and contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, the original features and more background information about the data cannot be provided. Features V1, V2, ..., V28 are the principal components obtained with PCA. The only features which have not been transformed with PCA are 'Time' and 'Amount'.

## How to Download the Dataset

1. Visit the Kaggle dataset page: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Sign in to your Kaggle account (or create one if you don't have it)
3. Click the "Download" button
4. Once downloaded, extract the ZIP file
5. Place the `creditcard.csv` file in this directory (`data/`)

## Alternative Method (Using Kaggle API)

If you have the Kaggle API set up, you can download the dataset using the following command:

```bash
kaggle datasets download mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
rm creditcardfraud.zip
```

To set up the Kaggle API:
1. Go to your Kaggle account settings
2. Scroll down to the API section and click "Create New API Token"
3. This will download a `kaggle.json` file
4. Place this file in `~/.kaggle/` directory
5. Run `chmod 600 ~/.kaggle/kaggle.json` to set proper permissions
6. Install the Kaggle API: `pip install kaggle`

## Data Preprocessing

The raw data will be preprocessed by the scripts in `src/data/`. The preprocessing steps include:
- Handling missing values (if any)
- Scaling the 'Amount' feature
- Splitting into training and testing sets
- Handling class imbalance using techniques like SMOTE

The preprocessed data will be saved in the `data/processed/` directory. 