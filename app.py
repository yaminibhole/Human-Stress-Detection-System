import os
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import google.generativeai as genai
from google.generativeai import chat

# Configure the API key and generative model
api_key = 'Your api key'# Enter you API KEY
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

generation_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config
)

# Load the pre-trained model
with open('model/stress_detection.pkl', 'rb') as file:
    reg = pickle.load(file)

# Prediction tab
def prediction_tab():
    st.title('Stress Level Predictor')

    st.markdown(
        """
        <style>
            .stApp {
                background-color: burlywood;
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
        \nWelcome to the Stress Level Predictor! Input physiological and environmental features to predict stress levels. Explore different scenarios, experiment with varying inputs, and gain insights into potential stress levels. Make informed decisions and take proactive measures to manage stress effectively in this interactive and exploratory Stress Level Predictor.
        """,
        unsafe_allow_html=True,
    )

    # Take user input for features
    st.sidebar.title("Enter Feature Values")
    selected_sr = st.sidebar.number_input("Snoring Rate:")
    selected_rr = st.sidebar.number_input("Respiration Rate:")
    selected_t = st.sidebar.number_input("Body Temperature:")
    selected_lm = st.sidebar.number_input("Limb Movement:")
    selected_bo = st.sidebar.number_input("Blood Oxygen Level:")
    selected_rem = st.sidebar.number_input("Rapid Eye Movement:")
    selected_sh = st.sidebar.number_input("Sleep Hours:")
    selected_hr = st.sidebar.number_input("Heart Rate:")

    # Create user input DataFrame
    user_input = pd.DataFrame({
        'sr': [selected_sr],
        'rr': [selected_rr],
        't': [selected_t],
        'lm': [selected_lm],
        'bo': [selected_bo],
        'rem': [selected_rem],
        'sh': [selected_sh],
        'hr': [selected_hr]
    })

    # Define function for prediction
    def predict_stress_level(user_input):
        user_pred = reg.predict(user_input)
        return user_pred

    # Create button for prediction
    if st.button("Predict"):
        # Get prediction
        prediction = predict_stress_level(user_input)

        # Define thresholds for stress levels
        threshold_low = 1.0
        threshold_medium_low = 2.0
        threshold_medium_high = 3.0
        threshold_high = 4.0

        # Determine stress level category
        if prediction < threshold_low:
            predicted_stress_level = 'Low/Normal'
        elif threshold_low <= prediction < threshold_medium_low:
            predicted_stress_level = 'Medium Low'
        elif threshold_medium_low <= prediction < threshold_medium_high:
            predicted_stress_level = 'Medium'
        elif threshold_medium_high <= prediction < threshold_high:
            predicted_stress_level = 'Medium High'
        else:
            predicted_stress_level = 'High'

        # Display prediction result
        st.subheader('Predicted Stress Level:')
        if predicted_stress_level == 'Low/Normal':
            st.success("Your stress level is low. Keep taking care of yourself! 🙂")
        elif predicted_stress_level == 'Medium Low':
            st.warning("Your stress level is moderate. Take some time to relax and unwind. 😐")
        elif predicted_stress_level == 'Medium':
            st.error("Your stress level is high! It's crucial to take steps to reduce it for your well-being! 😞")
        elif predicted_stress_level == 'Medium High':
            st.error("Your stress level is critical. Please prioritize your mental and physical health and seek support from loved ones or professionals!! 😫")
        else:
            st.success("Congratulations! Your stress level prediction is not available. You seem to be stress-free and calm 😄")

        st.markdown(
            """
            \nIf you'd like to talk to someone or need more personalized support, check out our Somebody tab. There, you'll find your AI assistant ready to chat and offer guidance. 😊
            """,
            unsafe_allow_html=True,
        )

# Somebody tab
def ai_interaction(prompt):
    try:
        # Check if there is an existing chat session
        if 'chat_session' not in st.session_state:
            # Start a new chat session
            chat_session = generation_model.start_chat()
            st.session_state.chat_session = chat_session
        else:
            chat_session = st.session_state.chat_session

        # Send user prompt and get response
        response = chat_session.send_message(prompt)

        # Extract the text from the response
        response_text = response.text.strip()

        # Append response to the history
        st.session_state.chat_history.append(f"Somebody: {response_text}")

        return response_text

    except Exception as e:
        return f"An error occurred: {e}"


def ai_interaction(prompt):
    try:
        # Check if there is an existing chat session
        if 'chat_session' not in st.session_state:
            # Start a new chat session
            chat_session = generation_model.start_chat()
            st.session_state.chat_session = chat_session
        else:
            chat_session = st.session_state.chat_session

        # Send user prompt and get response
        response = chat_session.send_message(prompt)

        # Extract the text from the response
        response_text = response.text.strip()

        # Append response to the history
        st.session_state.chat_history.append(f"Somebody: {response_text}")

        return response_text

    except Exception as e:
        st.session_state.chat_history.append(f"Somebody: An error occurred: {e}")
        return f"An error occurred: {e}"


def somebody_tab():
    st.title('Somebody: Your AI Assistant')

    st.markdown(
        """
        <style>
            body { }
            .stApp {
                background-color: burlywood;
                color: black;
                text-align: justify;
            }
            p {
                font-size: 18px;
            }
            .chat-message {
                font-size: 18px;
                margin: 5px 0;
            }
            .chat-label {
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        \nMeet Somebody, your AI assistant! There's somebody for everyone, and Somebody will be your friend and guide in managing stress and answering your questions. Engage in a friendly chat with Somebody, and get personalized advice and support to help you navigate through stressful situations.
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Chat input and interaction
    user_input = st.text_input("Type your message here:")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(f"You: {user_input}")
            ai_response = ai_interaction(user_input)

    # Display chat history
    st.subheader("Chat History:")
    for message in st.session_state.chat_history:
        if message.startswith("You:"):
            styled_message = f"<div class='chat-message'><span class='chat-label'>You:</span> {message[4:]}</div>"
        elif message.startswith("Somebody:"):
            styled_message = f"<div class='chat-message'><span class='chat-label'>Somebody:</span> {message[9:]}</div>"
        else:
            styled_message = f"<div class='chat-message'>{message}</div>"
        st.markdown(styled_message, unsafe_allow_html=True)

# About tab
def about_tab():
    st.title('About the Stress Detection Model')

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #E6C8BC;
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
        \nThe stress detection model employs a RandomForestClassifier algorithm trained on a dataset containing physiological and environmental features, along with corresponding stress levels. The model learns the underlying patterns and relationships within this data to predict stress levels based on input features.

        \nThe RandomForestClassifier is a powerful ensemble learning method used for classification tasks. It constructs multiple decision trees during training and outputs the most frequent class (mode) predicted by these trees. This technique enhances accuracy and robustness, making it particularly effective for complex tasks like predicting stress levels based on physiological data. Its ability to handle diverse features and reduce overfitting makes it an ideal choice for building reliable stress detection systems.

        \nThe prediction process involves inputting physiological and environmental features, such as snorring rate, respiration rate, temperature, limb movement, blood oxygenation, rapid eye movement, heart rate, and sleep duration. The model processes this input to generate a prediction for the stress level, allowing users to explore different scenarios and gain insights into potential stress levels.

        """,
        unsafe_allow_html=True,
    )

# Home tab
def home_tab():
    st.title('Welcome to Happify :)')

    st.markdown(
        """
        <style>
            .stApp {
                background-color: #ADDCF4;
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
        \nHappify is an interactive platform designed to predict stress levels based on physiological and environmental features. It provides users with insights into potential stress levels and enables them to explore different scenarios by inputting specific feature values.
        """,
        unsafe_allow_html=True,
    )

    # Display image
    img = Image.open('Images/stress.jpg')
    st.image(img, width=100, use_column_width=True)
    
    st.markdown(
        """
        \nBy providing a user-friendly and intuitive interface, Happify empowers individuals to take proactive measures towards managing their stress effectively, thereby promoting overall well-being and resilience.
        """,
        unsafe_allow_html=True,
    )

# Feedback tab
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
        \nI'd love to hear your thoughts and suggestions on Stress Detection System! Your feedback helps me improve and enhance the platform to better serve your needs. Whether it's a feature request, bug report, or general comment, we value your input and appreciate your contribution to making Stress Detection System even better.

        \nPlease share your feedback in the form below. Thank you !!
        """,
        unsafe_allow_html=True,
    )

    # Feedback form
    name = st.text_input("Name:", max_chars=50)
    email = st.text_input("Email:", max_chars=50)
    feedback = st.text_area("Please provide your feedback here:", height=200)
    if st.button("Submit Feedback"):
        # Process feedback (can be stored in a database or file)
        st.success(f"Thank you, {name}! We appreciate your feedback. We'll review your input and work towards improving Stress Detection Explorer.")

# Create tabs
tabs = ["Home", "About", "Stress Level Predictor", "Somebody", "Feedback"]
selected_tab = st.sidebar.radio("Welcome :)", tabs)

# Show the selected tab
if selected_tab == "Home":
    home_tab()
elif selected_tab == "About":
    about_tab()
elif selected_tab == "Feedback":
    feedback_tab()
elif selected_tab == "Somebody":
    somebody_tab()
else:
    prediction_tab()
