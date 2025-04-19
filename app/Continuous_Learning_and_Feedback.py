import streamlit as st
import pandas as pd
import os
import sys
import datetime
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)


def analyze_sentiments(feedback_df):
    feedback_df['Sentiment'] = feedback_df['Feedback Comments'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    feedback_df['Sentiment_Label'] = feedback_df['Sentiment'].apply(
        lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))
    return feedback_df


def cluster_feedback(feedback_df):
    # Dynamically set the number of clusters based on feedback size
    feedback_count = len(feedback_df)
    # Set a heuristic for number of clusters (e.g., 1 cluster per 10 feedbacks, minimum of 3 clusters)
    n_clusters = max(3, feedback_count // 10)

    if feedback_count < n_clusters:
        feedback_df['Cluster'] = 0  # fallback single cluster
        return feedback_df

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(feedback_df['Feedback Comments'].astype(str))
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    feedback_df['Cluster'] = model.labels_
    return feedback_df


def summarize_feedback(df):
    st.subheader("📋 Feedback Summary")
    st.write("Total Feedback:", len(df))
    st.write("Average Rating:", round(df['Feedback Rating'].mean(), 2))
    negative_comments = df[df['Sentiment_Label'] == 'Negative']['Feedback Comments'].astype(str).str.cat(sep=' ')
    common_issues = Counter(negative_comments.split()).most_common(5)
    st.write("Top Complaint Keywords:", [word for word, count in common_issues])
    st.dataframe(df)


def plot_sentiment_trend(df):
    st.subheader("📈 Sentiment Trend Over Time")
    if 'Timestamp' not in df.columns:
        st.warning("No timestamp data available for sentiment trend.")
        return

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_sorted = df.sort_values('Timestamp')

    fig, ax = plt.subplots()
    ax.plot(df_sorted['Timestamp'], df_sorted['Sentiment'], marker='o', linestyle='-', color='green')
    ax.set_title("Sentiment Polarity Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score")
    ax.axhline(0, color='gray', linestyle='--')
    st.pyplot(fig)


def plot_cluster_wordclouds(df):
    st.subheader("☁️ Clustered Word Clouds")

    if 'Cluster' not in df.columns:
        st.warning("No clustering information available.")
        return

    for cluster_id in sorted(df['Cluster'].unique()):
        st.markdown(f"**Cluster {cluster_id}**")
        text = ' '.join(df[df['Cluster'] == cluster_id]['Feedback Comments'].astype(str))
        if not text.strip():
            st.write("No content in this cluster.")
            continue
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)


def add_feedback_form(data_file_path):
    st.sidebar.header("➕ Add New Feedback")
    with st.sidebar.form("feedback_form"):
        user_name = st.text_input("Name")
        feedback_rating = st.slider("Rating (1 to 5)", 1, 5, 3)
        feedback_comments = st.text_area("Comments")
        feedback_type = st.selectbox("Feedback Type", ["Bug", "Feature Request", "General Feedback"])
        submitted = st.form_submit_button("Submit Feedback")

        if submitted and user_name and feedback_comments:
            new_feedback = pd.DataFrame({
                'Name': [user_name],
                'Feedback Rating': [feedback_rating],
                'Feedback Comments': [feedback_comments],
                'Feedback Type': [feedback_type],
                'Timestamp': [datetime.datetime.now()]
            })

            if os.path.exists(data_file_path):
                existing_data = pd.read_csv(data_file_path)
                is_duplicate = ((existing_data['Name'] == user_name) &
                                (existing_data['Feedback Comments'] == feedback_comments)).any()
                if not is_duplicate:
                    updated_data = pd.concat([existing_data, new_feedback], ignore_index=True)
                    updated_data.to_csv(data_file_path, index=False)
                    st.sidebar.success("✅ Feedback submitted successfully!")
                    st.session_state["feedback_submitted"] = True
                else:
                    st.sidebar.warning("⚠️ Similar feedback already exists.")
            else:
                new_feedback.to_csv(data_file_path, index=False)
                st.sidebar.success("✅ Feedback submitted successfully!")
                st.session_state["feedback_submitted"] = True

    if st.sidebar.button("🗑️ Delete All Feedback"):
        if os.path.exists(data_file_path):
            empty_df = pd.DataFrame(
                columns=['Name', 'Feedback Rating', 'Feedback Comments', 'Feedback Type', 'Timestamp'])
            empty_df.to_csv(data_file_path, index=False)
            st.sidebar.success("🧹 All feedback has been deleted!")


def continuous_learning_and_feedback():
    if "feedback_submitted" in st.session_state and st.session_state["feedback_submitted"]:
        st.rerun()

    st.title("🏡 Continuous Learning and Feedback Dashboard")

    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Feedback.csv')
    if not os.path.exists(data_file_path):
        st.warning("⚠️ Feedback data file not found.")
        return

    add_feedback_form(data_file_path)

    feedback_data = pd.read_csv(data_file_path)
    feedback_data = feedback_data.dropna(subset=['Feedback Rating', 'Feedback Comments'])

    if feedback_data.empty:
        st.info("No feedback entries available. Submit new feedback to begin analysis.")
        return

    feedback_data = analyze_sentiments(feedback_data)
    feedback_data = cluster_feedback(feedback_data)

    summarize_feedback(feedback_data)
    plot_sentiment_trend(feedback_data)
    plot_cluster_wordclouds(feedback_data)

    avg_rating = feedback_data["Feedback Rating"].mean()
    negative_feedback_count = len(feedback_data[feedback_data["Feedback Rating"] < 3])

    rating_threshold = 3.5
    negative_feedback_threshold = 20

    rating_percentage = avg_rating / rating_threshold
    negative_feedback_percentage = negative_feedback_count / negative_feedback_threshold

    st.write(f"Avg. Rating: {avg_rating:.2f}, Negative Feedback Count: {negative_feedback_count}")

    st.subheader("🔹 Alert Meter")
    col1, col2 = st.columns(2)
    with col1:
        st.progress(min(1.0, rating_percentage), text=f"Avg. Rating: {avg_rating:.2f}")
    with col2:
        st.progress(min(1.0, negative_feedback_percentage),
                    text=f"Negative Feedback: {negative_feedback_count}/{negative_feedback_threshold}")

    if rating_percentage >= 1.0 or negative_feedback_percentage >= 0.9:
        st.warning("⚠️ The system is approaching the alert threshold. Please review the user feedback.")
    else:
        st.success("✅ The system is performing well based on the user feedback.")


# Run app
if __name__ == "__main__":
    continuous_learning_and_feedback()
