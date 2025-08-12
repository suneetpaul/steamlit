import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.cluster import KMeans

def get_connection():
    return mysql.connector.connect(
        host="localhost",       
        user="root",            
        password="101003",
        database="customer_db"  
        
    )

def load_customers():
    
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM customer", conn)
    conn.close()
    return df

def add_customer(data):
   
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO customers 
        (age, gender, income, spending_score, membership_years, purchase_frequency, preferred_category, last_purchase_amount)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cursor.execute(query, data)
    conn.commit()
    conn.close()

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍 Customer Segmentation Tool")
st.write("Analyze and group customers using K-Means clustering.")

df = load_customers()

numeric_features = ["age", "income", "spending_score", "membership_years", "purchase_frequency", "last_purchase_amount"]

n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=3)

if not df.empty:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[numeric_features])
else:
    kmeans = None
    st.warning("No data in the database yet. Add customers to start clustering.")

st.subheader("➕ Add a New Customer")
with st.form("customer_form"):
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    income = st.number_input("Income", min_value=0.0, step=100.0)
    score = st.number_input("Spending Score (0-100)", min_value=0, max_value=100, step=1)
    years = st.number_input("Membership Years", min_value=0, step=1)
    freq = st.number_input("Purchase Frequency", min_value=0, step=1)
    category = st.text_input("Preferred Category")
    last_amt = st.number_input("Last Purchase Amount", min_value=0.0, step=0.01)
    submitted = st.form_submit_button("Save Customer")

if submitted:
  
    add_customer((age, gender, income, score, years, freq, category, last_amt))
    st.success("✅ Customer added successfully!")

    df = load_customers()

    if kmeans:
        cluster = kmeans.predict([[age, income, score, years, freq, last_amt]])[0]
        st.info(f"🧩 This customer belongs to Cluster {cluster}")


if kmeans and not df.empty:
    st.subheader("📊 Customer Segmentation Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df["income"],
        df["spending_score"],
        c=df["Cluster"],
        cmap="viridis"
    )
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation (Income vs Spending Score)")
    st.pyplot(fig)


if not df.empty:
    st.subheader("📄 Customer Data")
    st.dataframe(df)
