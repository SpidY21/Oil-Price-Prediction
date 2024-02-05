import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


def To_Be_Length(month, year):
    count = month + (year - 2024)
    return count


model = pickle.load(open("C:/Users/Yash/Desktop/ExcelR Project/App/RMSE_5_model.sav", 'rb'))


def forecasting(month, year):
    df = pd.read_excel(r"C:\Users\Yash\Desktop\ExcelR Project\Codes\RWTCm.xls")
    df.index.freq = 'MS'
    df.set_index('Date', inplace=True)
    train = df.iloc[:444]
    test = df.iloc[444:]
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_test = scaler.transform(test)
    n_input = 12
    n_features = 1
    month = int(month)
    year = int(year)
    test_predictions = []
    first_eval_batch = scaled_test[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(len(test) + To_Be_Length(month=month, year=year)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_predictions = scaler.inverse_transform(test_predictions)
    return true_predictions[-1][0]


def main():
    st.title('Oil Price Prediction')

    month = st.text_input('Enter the Month: ')
    year = st.text_input('Enter the Year: ')

    result = 0

    if st.button('Predict'):
        if 12 < int(month) < 1:
            st.error('Please enter valid month')
        else:
            result = forecasting(month, year)
    st.success(round(result, 2))
    st.write(round(result, 2), "$ Per Barrel")


if __name__ == '__main__':
    main()
