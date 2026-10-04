import pandas as pd
import numpy as np
import joblib
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_invoice_data():
    conn = sqlite3.connect('D:/CodeAIML/python/project/Invoice Intelligence Machine Learning/Data/inventory.db')
    query = """
    WITH df AS (
        SELECT p.PONumber, count(distinct p.Brand) as total_brands, sum(p.Quantity) as total_Quantity, sum(p.Dollars) as total_item_dollars, avg(julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_days_to_receive from purchases p group by p.PONumber
    )
    SELECT
        v.PONumber,
        v. Quantity AS invoice_quantity,
        v. Dollars AS invoice_dollars,
        v.Freight,
        (julianday(v.InvoiceDate) - julianday (v.PODate)) AS days_po_to_invoice,
        (julianday(v. PayDate) - julianday (v.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_Quantity,
        pa.total_item_dollars,
        pa.avg_days_to_receive

        FROM vendor_invoice v
        LEFT JOIN df pa
        ON v.PONumber = pa.PONumber
"""
    df=pd.read_sql_query(query, conn)
    conn.close()

    return df

def create_label(row):
    if (abs(row['invoice_dollars']-row['total_item_dollars'])>5):
        return 1
    if row['avg_days_to_receive']>10:
        return 1
    return 0

def apply_label(df):
    df['flag_invoice'] = df.apply(create_label, axis=1)
    return df

def split_data(df,features,target):
    x=df[features]
    y=df[target]

    return train_test_split(x, y, test_size=0.2, random_state=42)

def scale_data(x_train, x_test):
    Scaler = StandardScaler()
    x_train_scaled = Scaler.fit_transform(x_train)
    x_test_scaled = Scaler.transform(x_test)

    joblib.dump(Scaler,'../../models/scaler.pkl')
    return x_train_scaled, x_test_scaled
