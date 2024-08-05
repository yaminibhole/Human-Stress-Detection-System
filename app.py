import os
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import google.generativeai as genai

# Check if running in Streamlit Cloud
is_streamlit_cloud = 'STREAMLIT_SERVER' in os.environ

if is_streamlit_cloud:
    # Fetch API key from Streamlit secrets
    api_key = st.secrets["api_key"]
else:
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    # Fetch API key from environment variables
    api_key = os.getenv('API_KEY')

# Configure the generative AI model
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

# Function to generate personalized suggestions
def generate_personalized_suggestions(features):
    # Construct prompt for the generative AI model
    stress_level = features['stress_level']
    prompt = f"Given the stress level category '{stress_level}', provide short personalized recommendations and suggestions for stress management based on the following features: {features}."
    chat_session = generation_model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text.strip()
    
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

        # Generate personalized suggestions
        features = {
            'snoring_rate': selected_sr,
            'respiration_rate': selected_rr,
            'body_temperature': selected_t,
            'limb_movement': selected_lm,
            'blood_oxygen_level': selected_bo,
            'rapid_eye_movement': selected_rem,
            'sleep_hours': selected_sh,
            'heart_rate': selected_hr,
            'stress_level': predicted_stress_level
        }
        suggestions = generate_personalized_suggestions(features)

        # Display prediction result and suggestions
        st.subheader('Predicted Stress Level:')
        st.write(predicted_stress_level)
        
        st.subheader('Personalized Suggestions:')
        st.write(suggestions)

        st.markdown(
            """
            \nIf you'd like to talk to someone or need more personalized support, check out our Somebody tab. There, you'll find your AI assistant ready to chat and offer guidance. 😊
            """,
            unsafe_allow_html=True,
        )

# AI Interaction function
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

# Somebody tab
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
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        st.markdown(f"<p class='chat-message'>{message}</p>", unsafe_allow_html=True)

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
        # Process feedback (can be stored in a database or file)
        st.success(f"Thank you, {name}! We appreciate your feedback. We'll review your input and work towards improving Stress Detection Explorer.")


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
        \nOur stress detection model utilizes a RandomForestClassifier, trained on physiological and environmental features to predict stress levels. Key features include:
        - Snoring Rate
        - Respiration Rate
        - Body Temperature
        - Limb Movement
        - Blood Oxygen Level
        - Rapid Eye Movement
        - Sleep Hours
        - Heart Rate

        \nThe RandomForestClassifier is an ensemble learning method that builds multiple decision trees and aggregates their results for accurate predictions. This model helps identify stress levels categorized as Low/Normal, Medium Low, Medium, Medium High, and High, providing valuable insights for effective stress management.

        \nExplore our tool to understand and manage your stress levels better. Your well-being is our priority!
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
        \nNavigate through the tabs to explore features:
        \n- **Home:** Get introduced to the system and view the welcome image.
        \n- **About:** Learn more about how the stress detection model works.
        \n- **Prediction:** Input your data to get stress level predictions and personalized suggestions.
        \n- **Somebody:** Chat with the AI assistant for personalized support and guidance.
        \n- **Feedback:** Share your thoughts and provide feedback to help us improve the system.

        \nStart exploring and take control of your stress management today!
        """,
        unsafe_allow_html=True,
    )

# Main application
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "About", "Prediction", "Somebody", "Feedback"])

    if selection == "Home":
        home_tab()
    elif selection == "About":
        about_tab()
    elif selection == "Prediction":
        prediction_tab()
    elif selection == "Somebody":
        somebody_tab()
    elif selection == "Feedback":
        feedback_tab()

if __name__ == "__main__":
    main()
