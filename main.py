import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image

# Load data
df = pd.read_csv('dataset.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Features and target variable
X = df.drop(['sl'], axis=1)
Y = df['sl']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Train the model
reg = RandomForestRegressor()
reg.fit(X_train, Y_train)
pred = reg.predict(X_test)
score = r2_score(Y_test, pred)

# Prediction tab
def prediction_tab():
    st.title('Stress Level Predictor')

    st.markdown(
        """
        <style>
            body { }
            .stApp {
                background-color: burlywood;
                color: black;
                text-align:justify;
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

        # Convert predicted stress level to categorical
        stress_level_categories = {0: 'Low/Normal', 1: 'Medium Low', 2: 'Medium', 3: 'Medium High', 4: 'High'}
        predicted_stress_level = stress_level_categories[int(round(prediction[0]))]

        # Display prediction result
        st.subheader('Predicted Stress Level:')
        if prediction == 1:
            st.success("Your stress level is low. Keep taking care of yourself! 🙂")
            st.write("Stress score =", str(prediction[0]))
            st.write("Congrats on maintaining a low stress level! Keep up the good work by practicing mindfulness, staying physically active, and nurturing your relationships. Make time for hobbies and self-care to sustain your well-being.")
        elif prediction == 2:
            st.warning("Your stress level is moderate. Take some time to relax and unwind. 😐")
            st.write("Stress score =", str(prediction[0]))
            st.write("While your stress level is moderate, it's essential to manage it proactively. Identify stress triggers, practice relaxation techniques, and seek support from loved ones or professionals if needed. Prioritize self-care and balance responsibilities with downtime.")
        elif prediction == 3:
            st.error("Your stress level is high! It's crucial to take steps to reduce it for your well-being! 😞")
            st.write("Stress score =", str(prediction[0]))
            st.write("Your high stress level requires immediate attention. Practice stress reduction techniques, establish healthy boundaries, and consider seeking professional help to address underlying stressors. Prioritize self-care and prioritize your well-being.")
        elif prediction == 4:
            st.error("Your stress level is critical. Please prioritize your mental and physical health and seek support from loved ones or professionals!! 😫")
            st.write("Stress score =", str(prediction[0]))
            st.write("Your very high stress level demands comprehensive intervention. Seek professional support, prioritize self-care practices, and lean on your support network for assistance. Remember that your well-being is a priority, and it's okay to ask for help.")
        else:
            st.success("Congratulations! Your stress level prediction is not available. You seem to be stress-free and calm 😄")
            st.write("Must be nice being stress-free, sipping on your imaginary margaritas while the rest of us are stuck in the chaos. Enjoy your stress-free bubble – just don't burst it, or you might have to join the rest of us in the real world! Bwahahhaha")

# About tab
def about_tab():
    st.title('About the Stress Detection Model')

    st.markdown(
        """
        <style>
            body {
                
            }
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
        \nThe stress detection model employs a RandomForestRegressor algorithm trained on a dataset containing physiological and environmental features, along with corresponding stress levels. The model learns the underlying patterns and relationships within this data to predict stress levels based on input features.

        \nThe RandomForestRegressor is an ensemble learning method known for its versatility and accuracy in regression tasks. It constructs multiple decision trees during training and outputs the average prediction of the individual trees, resulting in robust and reliable predictions. The model's ability to capture complex relationships in data makes it well-suited for predicting stress levels influenced by various factors.

        \nThe prediction process involves inputting physiological and environmental features, such as skin resistance, respiration rate, temperature, limb movement, blood oxygenation, rapid eye movement, heart rate, and sleep duration. The model processes this input to generate a prediction for the stress level, allowing users to explore different scenarios and gain insights into potential stress levels.

        """,
        unsafe_allow_html=True,
    )
    # Display model performance
    st.subheader('Using Random Forest Regressor')
    st.write(df)
    st.subheader('Model Performance:')
    st.write(score)
    
# Home tab
def home_tab():
    st.title('Welcome to Happify :)')

    st.markdown(
        """
        <style>
            body { }
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
    img = Image.open('stress.jpg')
    st.image(img, width=150, use_column_width=True)

    st.markdown(
        """
        \nStress is a natural response to the demands and pressures of daily life, affecting individuals both physically and mentally. While a certain level of stress can be motivating and adaptive, chronic or excessive stress can have detrimental effects on health and well-being. It can manifest in various ways, including physiological symptoms like increased heart rate, muscle tension, and compromised immune function, as well as psychological symptoms such as anxiety, irritability, and difficulty concentrating. Understanding and managing stress is essential for maintaining overall health and resilience. Happify aims to empower users by providing valuable insights into their stress levels, enabling them to take proactive steps towards stress management and promoting a healthier and more balanced lifestyle.
        \nLeveraging the capabilities of advanced machine learning techniques, Happify utilizes a RandomForestRegressor algorithm to generate accurate predictions for stress levels. By allowing users to input specific physiological and environmental parameters such as skin resistance, respiration rate, temperature, limb movement, blood oxygenation, rapid eye movement, heart rate, and sleep duration, Happify delivers personalized predictions tailored to each individual's unique circumstances.
        \nThrough Happify, users can not only explore their stress levels but also analyze trends and anticipate fluctuations with confidence and insight. By providing a user-friendly and intuitive interface, Happify empowers individuals to take proactive measures towards managing their stress effectively, thereby promoting overall well-being and resilience.
        """,
        unsafe_allow_html=True,
    )
# Feedback tab
def feedback_tab():
    st.title('Share Your Thoughts')

    st.markdown(
        """
        <style>
            body { }
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
        st.success("Thank you, {}! We appreciate your feedback. We'll review your input and work towards improving Stress Detection Explorer.".format(name))

# Create tabs
tabs = ["Home", "About", "Stress Level Predictor", "Feedback"]
selected_tab = st.sidebar.radio("Welcome :)", tabs)

# Show the selected tab
if selected_tab == "Home":
    home_tab()
elif selected_tab == "About":
    about_tab()
elif selected_tab == "Feedback":
    feedback_tab()
else:
    prediction_tab()
