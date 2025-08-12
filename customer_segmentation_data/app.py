import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your dataset
DATA_PATH = "D:\project3\data\customer_segmentation_data.csv"
df = pd.read_csv(DATA_PATH)

# Select numeric features for clustering
features = ["age", "income", "spending_score", "membership_years", "purchase_frequency", "last_purchase_amount"]

# Sidebar: choose number of clusters
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

# Train KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[features])

st.write(df)


# Title
st.title("Customer Segmentation Tool")

# Form for new customer
with st.form("customer_form"):
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    income = st.number_input("Income", min_value=0, step=100)
    score = st.number_input("Spending Score (0-100)", min_value=0, max_value=100, step=1)
    years = st.number_input("Membership Years", min_value=0, step=1)
    freq = st.number_input("Purchase Frequency", min_value=0, step=1)
    last_amt = st.number_input("Last Purchase Amount", min_value=0.0, step=0.01)
    submitted = st.form_submit_button("Submit")

if submitted:
    new_customer = pd.DataFrame([[age, income, score, years, freq, last_amt]],
                                 columns=features)
    cluster = kmeans.predict(new_customer)[0]
    st.success(f"Customer belongs to Cluster {cluster}")

# Matplotlib scatter plot (income vs spending_score)
fig, ax = plt.subplots()
scatter = ax.scatter(df["income"], df["spending_score"], c=df["Cluster"], cmap="viridis")
ax.set_xlabel("Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation (Income vs Spending Score)")
st.pyplot(fig)
