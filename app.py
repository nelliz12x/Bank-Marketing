import streamlit as st
import numpy as np
import pandas as pd
import pypickle
from sklearn.preprocessing import LabelEncoder

load_model = pypickle.load("model.pkl")


def prediction(data):

    label = LabelEncoder()
    df = pd.DataFrame(data)

    lab = [1, 2, 3, 4, 6, 7, 8, 10, 15]
    for i in lab:
        df.iloc[i] = label.fit_transform(df.iloc[i])

    num_data = df.values.reshape(1, -1)

    pred = load_model.predict(num_data)

    if pred[0] == 1:
        return "The Customer will made a deposit"
    else:
        return "The Customer will not make a deposit"
    

def main():
    st.title("Bank Marketing Predictive Model")
    age = st.number_input("Age of the customer: ")
    job = st.text_input("Client's Occupation: ")
    marital = st.text_input("Client's Marital Status : divorced, married, single")
    education = st.text_input("Academic Qualification: ")
    default = st.text_input("Client has credit in default?: Yes, No")
    balance = st.number_input("Balance Paid: ")
    housing = st.text_input("Client has housing loan?: Yes, No")
    loan = st.text_input("Client has personal loan?: Yes, No")
    contact = st.text_input("Client contact communication type: cellular, telephone")
    day = st.text_input("Last contact day of the week: ")
    month = st.text_input("Last contact month of year: ")
    duration = st.number_input("Last contact duration")
    campaign = st.number_input("Number of contacts performed during this campaign and for this client: ")
    pday = st.number_input("Number of days that passed by after the client was last contacted from a previous campaign: ")
    previous = st.number_input("Number of contacts performed before this campaign and for this client: ")
    poutcome = st.text_input("Outcome of the previous marketing campaign")

    Deposit = ""

    if st.button("Result"):
        Deposit = prediction([age, job, marital, education, default, balance, housing,
       loan, contact, day, month, duration, campaign, pday, previous, poutcome])
        
    st.success(Deposit)


if __name__ == "__main__":
    main()