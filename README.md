# Human-Stress-Detection-System
The Human-Stress-Detection-System is an interactive platform that predicts stress levels using a RandomForestClassifier algorithm. Users input physiological data such as snoring rate, respiration rate, and heart rate to get personalized stress predictions. The system also provides tailored suggestions for stress management. Additionally, it features Somebody, an AI assistant that offers real-time support and guidance through interactive chat, enhancing the overall user experience in managing stress.The system also includes an admin panel for managing feedback stored in an SQLite database.

## Project Structure

The project directory is organized as follows:

- data/: Contains dataset files (e.g., dataset.csv).
- images/: Contains image assets used in the application (e.g., stress.jpg).
- model/: Contains the pre-trained model file (stress_detection.pkl) and the Jupyter notebook (stress_detection.ipynb).
- visualizations/: Contains visualizations of important features and data insights.
- app.py: Main application script for the Streamlit app.
- feedback.py: Script for handling feedback.
- requirements.txt: Lists the Python packages required to run the app.
- README.md: Project documentation.

# Features
The Stress Detection sysytem offers several key features:

1. Stress Level Prediction: Predicts stress levels based on user-provided data.
2. Personalized Recommendations: Provides stress management advice tailored to the user.
3. AI Assistant: Engages with users for interactive support.
4. Visualizations: Displays key data insights and feature importance.
5. Feedback System: Collects and stores feedback in an SQLite database.
6. Admin Panel: Manages and reviews user feedback.

## Prerequisites
Before running the project, ensure you have the following installed:
- pandas
- streamlit
- Pillow
- google-generativeai
- matplotlib
- scikit-learn
- python-dotenv
- sqlite3

## Usage
Run the App: streamlit run app.py

Interact with the app:
- Home: Navigate to the home tab to view an overview of the application.
- About: Find information about the dataset, model accuracy, and other details.
- Visualization: Explore visualizations of important features and data insights.
- Predict: Enter feature values to get stress level predictions, along with personalized recommendations and suggestions.
- Somebody: Engage with the AI Assistant for interactive and personalized support.
- Feedback: Provide feedback on your experience and review feedback submissions.
- Admin: Manage and review user feedback through the admin panel (restricted access).

## Contact
For questions or suggestions, please contact yaminibhole20@gmail.com.
