import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# تحميل البيانات
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# ======== Content-Based Filtering ========
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ======== Collaborative Filtering باستخدام SVD من scipy ========
def compute_svd_predictions(ratings_df):
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    matrix = user_item_matrix.values

    # تطبيق SVD
    U, sigma, Vt = svds(matrix, k=20)
    sigma = np.diag(sigma)
    preds = np.dot(np.dot(U, sigma),Vt)
    pred_df = pd.DataFrame(preds, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return pred_df

predicted_ratings_df = compute_svd_predictions(ratings)

# ======== دالة التوصية الهجينة ========
def hybrid_recommendation(user_id, top_n=10):
    if user_id not in predicted_ratings_df.index:
        return None

    user_preds = predicted_ratings_df.loc[user_id]
    user_history = ratings[ratings.userId == user_id]
    unseen_movies = movies[~movies["movieId"].isin(user_history["movieId"])]
    unseen_movies = unseen_movies.copy()
    unseen_movies["predicted_rating"] = unseen_movies["movieId"].map(user_preds)

    # Content-based score
    cb_scores = []
    for movie_id in unseen_movies["movieId"]:
        try:
            idx = movies[movies["movieId"] == movie_id].index[0]
            score = cosine_sim[idx].mean()
        except:
            score = 0
        cb_scores.append(score)

    unseen_movies["cb_score"] = cb_scores
    unseen_movies["hybrid_score"] = (0.7 * unseen_movies["predicted_rating"]) + (0.3 * unseen_movies["cb_score"])

    recommendations = unseen_movies.sort_values("hybrid_score", ascending=False).head(top_n)
    return recommendations[["title", "genres", "hybrid_score"]]

# ======== واجهة المستخدم ========
st.title("🎬 Hybrid Movie Recommender ")
st.write("احصل على توصيات أفلام مخصصة باستخدام نموذج هجين.")

user_id = st.number_input("ادخل رقم المستخدم (userId):", min_value=1, value=1, step=1)

if st.button("اعرض الأفلام المقترحة"):
    result = hybrid_recommendation(user_id)
    if result is None or result.empty:
        st.error("هذا المستخدم غير موجود أو لا توجد بيانات كافية.")
    else:
        for _, row in result.iterrows():
            st.subheader(row["title"])
            st.write(f"🎭 type: {row['genres']}")
            st.write(f"🔮 recommendation degree **{round(row['hybrid_score'], 2)}**")
