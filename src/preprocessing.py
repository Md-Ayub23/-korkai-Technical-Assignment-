
import pandas as pd


def load_data(file_path):
    """Load Dataset"""
    return pd.read_csv(file_path, sep="\t")


def clean_data(df):
    """Perform data cleaning."""

    df["Income"] = df["Income"].fillna(df["Income"].median())

    df = df[df["Income"] != 666666].copy()


    income_cap = df["Income"].quantile(0.99)
    df["Income"] = df["Income"].clip(upper=income_cap)

    df = df[df["Year_Birth"] >= 1940].copy()

    df = df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])

    rare_status = ["YOLO", "Absurd", "Alone"]
    df["Marital_Status"] = df["Marital_Status"].replace(rare_status, "Other")

    return df


def feature_engineering(df):
    """Create new features using a FIXED reference date so results are reproducible
    and tenure isn't artificially inflated by however many years have passed
    since this dataset was collected."""

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

   
    reference_date = df["Dt_Customer"].max() + pd.Timedelta(days=1)
    reference_year = reference_date.year

    df["Age"] = reference_year - df["Year_Birth"]
    df["Customer_Tenure"] = (reference_date - df["Dt_Customer"]).dt.days
    df = df.drop(columns=["Dt_Customer"])

    df["Previous_Purchase"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )

    df["Total_Spending"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    df["Children"] = df["Kidhome"] + df["Teenhome"]

   
    df["TotalAcceptedCmp"] = (
        df["AcceptedCmp1"]
        + df["AcceptedCmp2"]
        + df["AcceptedCmp3"]
        + df["AcceptedCmp4"]
        + df["AcceptedCmp5"]
    )

    return df


def preprocess(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    return df


if __name__ == "__main__":
    df = preprocess("../data/marketing_campaign.csv")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)