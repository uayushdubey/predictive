import streamlit as st
import pandas as pd
import os
import sys
import datetime
import logging
import smtplib
from collections import Counter
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

def send_alert(avg_rating, rating_threshold, negative_feedback_count, negative_feedback_threshold):
    sender_email = "app.technicalteam@gmail.com"
    receiver_email = "usaksham01@gmail.com"
    password = os.environ.get('EMAIL_PASSWORD')
    subject = f"User Feedback Alert - System Approaching Thresholds"

    body = f"""
    Dear Engineering Team,

    Our user feedback monitoring system has detected that one or more of the configured thresholds has been approached or exceeded.

    1. Average User Rating Threshold:
      ->Current Average Rating: {avg_rating}
      ->Threshold: {rating_threshold}

    2. Negative Feedback Threshold:
      ->Current Negative Feedback Count: {negative_feedback_count}
      ->Threshold: {negative_feedback_threshold}

    Please review the user feedback data and submit an action plan within the next 24 hours.

    Best regards,
    The User Feedback Monitoring System
    """

    attachment_path = os.path.join(root_dir, "Component_datasets", "Feedback.csv")
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with open(attachment_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
        message.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(message)
            logging.info("E-mail alert sent successfully")
            st.success(f"Alert report email ✉️ has been sent to {receiver_email}.")
            st.warning("📬 Haven't received the email invitation? Check your spam folder!")
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        st.error("An error occurred while sending the email alerts.")

def send_feedback_session_invitation(session_date, session_time, email_addresses):
    sender_email = "app.technicalteam@gmail.com"
    password = os.environ.get('EMAIL_PASSWORD')

    for email_address in email_addresses:
        subject = f"Invitation: Predictive Guardians Feedback Session on {session_date.strftime('%B %d, %Y')} at {session_time.strftime('%I:%M %p')}"
        body = f"""
        Dear Stakeholder,

        You are invited to the Predictive Guardians Feedback Session on {session_date} at {session_time}.

        Please let us know if you can attend the session.

        Best regards,
        The Predictive Guardians Team
        """

        attachment_path = os.path.join(root_dir, "Component_datasets", "Feedback.csv")
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email_address
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            message.attach(part)

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(message)
                logging.info("E-mail invitation sent successfully")
                st.success(f"Feedback session invitation email ✉️ has been sent to {email_address}.")
                st.warning("📬 Haven't received the email invitation? Check your spam folder!")
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            st.error("An error occurred while sending the email invitation.")

def analyze_sentiments(feedback_df):
    feedback_df['Sentiment'] = feedback_df['Feedback Comments'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    feedback_df['Sentiment_Label'] = feedback_df['Sentiment'].apply(lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral'))
    return feedback_df

def cluster_feedback(feedback_df, n_clusters=3):
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
                updated_data = pd.concat([existing_data, new_feedback], ignore_index=True)
            else:
                updated_data = new_feedback

            updated_data.to_csv(data_file_path, index=False)
            st.sidebar.success("✅ Feedback submitted successfully!")
            st.session_state["feedback_submitted"] = True  # Store the state of the feedback submission

def continuous_learning_and_feedback():
    if "feedback_submitted" in st.session_state and st.session_state["feedback_submitted"]:
        st.experimental_rerun()  # Trigger a rerun to reflect new feedback data

    st.title("🏡 Continuous Learning and Feedback Dashboard")

    data_file_path = os.path.join(root_dir, 'Component_datasets', 'Feedback.csv')
    if not os.path.exists(data_file_path):
        st.warning("⚠️ Feedback data file not found.")
        return

    add_feedback_form(data_file_path)

    feedback_data = pd.read_csv(data_file_path)

    feedback_data = analyze_sentiments(feedback_data)
    feedback_data = cluster_feedback(feedback_data)

    summarize_feedback(feedback_data)

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
        st.progress(rating_percentage, text=f"Avg. Rating: {avg_rating:.2f}")
    with col2:
        st.progress(negative_feedback_percentage, text=f"Negative Feedback: {negative_feedback_count}/{negative_feedback_threshold}")

    if rating_percentage >= 1.0 or negative_feedback_percentage >= 0.9:
        st.warning("The system is approaching the alert threshold. Please review the user feedback.")
        send_alert(avg_rating, rating_threshold, negative_feedback_count, negative_feedback_threshold)
    else:
        st.success("The system is performing well based on the user feedback.")

# Initialize app
if __name__ == "__main__":
    continuous_learning_and_feedback()
