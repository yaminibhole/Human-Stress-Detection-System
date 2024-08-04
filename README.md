# Human-Stress-Detection-System
The Human-Stress-Detection-System is an interactive platform that predicts stress levels using a RandomForestClassifier algorithm. Users input physiological data such as snoring rate, respiration rate, and heart rate to get personalized stress predictions. The system also provides tailored suggestions for stress management. Additionally, it features Somebody, an AI assistant that offers real-time support and guidance through interactive chat, enhancing the overall user experience in managing stress.

## Project Structure

- app.py: Main application script for the Streamlit app.
- model/: Contains the pre-trained model file (stress_detection.pkl) and the Jupyter notebook (stress_detection.ipynb).
- images/: Contains image assets used in the application (stress.jpg).
- requirements.txt: Lists the Python packages required to run the app.

# Features
The Stress Detection sysytem offers several key features:

1. Predict Stress Levels: Users can input physiological and environmental data to receive predictions about their stress levels.
2. Explore Scenarios: The platform allows users to experiment with different feature values to explore various stress scenarios.
3. AI Assistant: Somebody provides real-time support and personalized guidance through interactive chat, helping users manage stress effectively.
4. Feedback System: Users can provide feedback and suggestions for improving the platform.

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Pillow
- google-generativeai

## Usage
Run the App: streamlit run app.py

Interact with the App:
- Navigate to the home tab to view an overview of the app.
- Use the Stress Level Predictor tab to input feature values and get predictions.
- Engage with the AI Assistant in the "Somebody" tab for personalized support.
- Provide feedback in the Feedback tab.

## Contact
For questions or suggestions, please contact yaminibhole20@gmail.com.
