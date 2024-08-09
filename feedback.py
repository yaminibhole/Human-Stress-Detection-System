import streamlit as st
import sqlite3
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            feedback TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Save feedback to the SQLite database
def save_feedback_to_db(name, email, feedback):
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (name, email, feedback) VALUES (?, ?, ?)
    ''', (name, email, feedback))
    conn.commit()
    conn.close()

# View feedback after verifying the admin password
def view_feedback(admin_password):
    # Hash the password for comparison
    hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()
    stored_password_hash = os.getenv("ADMIN_PASSWORD_HASH")

    if hashed_password == stored_password_hash:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM feedback')
        rows = cursor.fetchall()
        conn.close()
        
        st.write("**Feedback List:**")
        for row in rows:
            st.write(f"ID: {row[0]}, Name: {row[1]}, Email: {row[2]}, Feedback: {row[3]}")
    else:
        st.error("Incorrect password. Access denied.")


# Feedback tab for users to submit their feedback
def feedback_tab():
    st.title('Share Your Thoughts')

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #DFEAF0;
                color: black;
                text-align: justify;
            }
            p {
                font-size: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        \nWe'd love to hear your thoughts and suggestions on Stress Detection Explorer! Your feedback helps us improve and enhance the platform to better serve your needs. Whether it's a feature request, bug report, or general comment, we value your input and appreciate your contribution to making Stress Detection Explorer even better.

        \nPlease share your feedback in the form below. Thank you for helping us create a more user-friendly and effective Stress Detection Explorer!
        """,
        unsafe_allow_html=True,
    )

    # Feedback form
    name = st.text_input("Name:", max_chars=50)
    email = st.text_input("Email:", max_chars=50)
    feedback = st.text_area("Please provide your feedback here:", height=200)
    
    if st.button("Submit Feedback"):
        if name and email and feedback:
            save_feedback_to_db(name, email, feedback)
            st.success(f"Thank you, {name}! We appreciate your feedback. We'll review your input and work towards improving Stress Detection Explorer.")
        else:
            st.error("Please fill out all fields before submitting.")

# Admin tab for viewing feedback (password protected)
def admin_tab():
    st.title('Admin Panel')
    password = st.text_input("Enter Admin Password:", type='password')
    
    if st.button("View Feedback"):
        if password:
            view_feedback(password)
        else:
            st.error("Please enter the admin password.")

# Run the database initialization
init_db()
