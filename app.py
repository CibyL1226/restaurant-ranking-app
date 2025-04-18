import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import ast
import streamlit as st

# Load model + embedder once
@st.cache_resource
def load_models():
    model = xgb.Booster()
    model.load_model("lambdamart_ranking_model(1).json")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embedder

# Load dataset once
@st.cache_data
def load_data():
    df = pd.read_csv("resturant_table.csv", compression = 'zip')
    df["text"] = df.apply(lambda row: f"{row['name']} {row['categories_title']} {row['city']}", axis=1)
    unique_texts = df["text"].unique()
    vector_map = {t: embedder.encode(t, convert_to_numpy=True) for t in unique_texts}
    df["vector"] = df["text"].map(vector_map)
    return df

def text_to_vec(text):
    return embedder.encode(text, convert_to_numpy=True)

# Feature engineering functions
def match_transaction(row, query):
    try:
        txns = ast.literal_eval(row["transactions"])
    except:
        txns = []
    return int(any(t in query.lower() for t in txns))

def category_match_score(row, query):
    return sum(1 for word in query.lower().split() if word in str(row["categories_title"]).lower())

def city_match(row, query):
    return int(row["city"].lower() in query.lower())

def open_overnight_score(row, query):
    return int(("overnight" in query.lower()) and (str(row["is_overnight"]).lower() == "true"))

# Ranking logic
def rank_query(query, model, restaurant_df):
    query_vec = text_to_vec(query)

    restaurant_df["similarity"] = restaurant_df["vector"].apply(lambda vec: 1 - cosine(query_vec, vec))
    restaurant_df["log_review_count"] = np.log1p(restaurant_df["review_count"])
    restaurant_df["price_score"] = restaurant_df["price"].apply(lambda p: 1 if 1 <= p <= 3 else 0)
    restaurant_df["transaction_match_score"] = restaurant_df.apply(lambda r: match_transaction(r, query), axis=1)
    restaurant_df["category_match_score"] = restaurant_df.apply(lambda r: category_match_score(r, query), axis=1)
    restaurant_df["city_match"] = restaurant_df.apply(lambda r: city_match(r, query), axis=1)
    restaurant_df["open_overnight_score"] = restaurant_df.apply(lambda r: open_overnight_score(r, query), axis=1)

    features = [
        "similarity", "rating", "log_review_count",
        "city_match", "category_match_score", "transaction_match_score",
        "open_overnight_score", "price_score"
    ]
    
    dmatrix = xgb.DMatrix(restaurant_df[features])
    restaurant_df["predicted_score"] = model.predict(dmatrix)

    top_results = (
        restaurant_df
        .sort_values(by="predicted_score", ascending=False)
        .drop_duplicates(subset="name")
        .head(5)
    )
    return top_results

# --- Streamlit UI ---
st.title("🍽️ Restaurant Ranking Search")
st.markdown("Type in a restaurant search query and get the top 5 ranked results.")

query = st.text_input("Enter your query:", "show me cafes that open late in new york")

if query:
    model, embedder = load_models()
    restaurant_df = load_data()
    with st.spinner("Ranking restaurants..."):
        results = rank_query(query, model, restaurant_df)

    st.subheader("🔝 Top 5 Results")
    for i, row in results.iterrows():
        st.image(row["image_url"], width=350, caption=row["name"])  # ✅ Show restaurant image
        st.markdown(f"**{row['name']}** - {row['city']}, {row['state']}")
        st.markdown(f"- ☎️ Phone: {row['display_phone']}")
        st.markdown(f"- ⭐️ Rating: {row['rating']} | 💬 Reviews: {row['review_count']}")
        st.markdown(f"- 📍 Address: {row['display_address']}")
        st.markdown(f"- 🌐 [Website]({row['url']})")
        st.markdown("---")
