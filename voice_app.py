
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

train = pd.read_csv('Training.csv')
test = pd.read_csv('Testing.csv')

test = test.drop(['Unnamed: 133'], axis=1)  # droping last column

test = pd.DataFrame(test)
train = pd.DataFrame(train)

train['Encephalitis'] = (train['high_fever'] | train['mild_fever'] | train['headache'] | train['cough']  | train['sweating']).astype(int)
train = train.drop(['high_fever', 'mild_fever', 'headache', 'cough', 'sweating'], axis=1)
other_columns = [col for col in train.columns if col not in ['Encephalitis', 'prognosis']]
column_order = other_columns + ['Encephalitis','prognosis']
train = train[column_order]

test = test.rename(columns=lambda x: x.replace('_', ' '))
train = train.rename(columns=lambda x: x.replace('_', ' '))

# splitting
y = train['prognosis']  # target
x = train.drop(['prognosis'], axis=1)  # symptoms
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)

rf = RandomForestClassifier(n_estimators=50, n_jobs=5, random_state=33, criterion="entropy")
clf_rf = rf.fit(x_train, y_train)
clf_rf.score(x_test, y_test)

# import openai
import pyttsx3
import openai
import streamlit as st

openai.api_key = "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"
# functions

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def Predict_disease(symptoms):
    if len(symptoms) > 1:
        # Create empty dataframe to hold symptoms
        symptoms_df = pd.DataFrame(columns=x_train.columns)
        # Add input symptoms to dataframe
        symptoms_df.loc[0] = 0
        for symptom in symptoms:
            symptoms_df[symptom] = 1

        # Make prediction using trained model
        pred = rf.predict(symptoms_df)
        return pred[0]
    else:
        return symptoms[0]


def main():
    st.title("Disease Predictor")
    # Take input of the Patient Details
    first_name = st.text_input("Enter First Name:")
    last_name = st.text_input("Enter Last Name:")
    phone_number = st.text_input("Enter Phone Number:")
    selected_date = st.date_input("Select the Date:")

    st.sidebar.header("Symptoms")
    options = train.drop('prognosis', axis=1).columns.tolist()
    symptoms = st.sidebar.multiselect("Select symptoms", options)
    if st.sidebar.button("Predict"):
        if symptoms and first_name and last_name and phone_number and selected_date:

            # Display input data
            st.subheader("Patient Information and Symptoms:")

            st.text(f"First Name: {first_name}")
            st.text(f"Last Name: {last_name}")
            st.text(f"Phone Number: {phone_number}")
            st.text(f"Selected Date: {selected_date}")
            st.text(f"Symptoms: {', '.join(symptoms)}")

            predicted_disease = Predict_disease(symptoms)
            st.success(f"Predicted Disease:\t{predicted_disease}")
            speak_text(predicted_disease)

if __name__ == "__main__":
    main()