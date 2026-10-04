import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_data(db_path: str):
    """Load vendor invoice data from SQLite database."""

    print("\nOpening database:")
    print(db_path)

    conn = sqlite3.connect(db_path)

    try:
        # Check which tables actually exist
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name;
            """,
            conn
        )

        print("\nTables found in this database:")
        print(tables)

        vendor_df = pd.read_sql_query(
            "SELECT * FROM vendor_invoice",
            conn
        )

    finally:
        conn.close()

    return vendor_df



def prepare_features(vendor_df: pd.DataFrame):
    """Prepare features and target."""

    # Remove rows where required values are missing
    vendor_df = vendor_df.dropna(subset=["Quantity", "Dollars", "Freight"])

    # Features
    x = vendor_df[["Quantity", "Dollars"]]

    # Target
    y = vendor_df["Freight"]

    return x, y


def split_data(x, y, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets."""

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test
