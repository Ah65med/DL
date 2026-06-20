import pandas as pd
from sqlalchemy import create_engine
import os
from schemas import Base, TransactionTable

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/anomaly_detection")

def ingest_data(csv_path: str):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    print(f"Data shape: {df.shape}")
    
    engine = create_engine(DATABASE_URL)
    
    print("Ensuring database schema...")
    Base.metadata.create_all(engine)
    
    print("Uploading to database (this may take a while)...")
    
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(str).str.replace('"', '')

    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    try:
        df.to_sql('transactions', engine, if_exists='append', index=False, method='multi', chunksize=1000)
        print("Ingestion complete.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    csv_path = "data/creditcard.csv"
    ingest_data(csv_path)
