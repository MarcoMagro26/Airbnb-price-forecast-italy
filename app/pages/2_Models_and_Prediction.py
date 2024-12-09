import time
import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, make_scorer, mean_absolute_percentage_error, median_absolute_error, PredictionErrorDisplay
from app.custom.custom_functions import random_color_generator

random.seed(874631)

st.set_page_config(page_title="Prediction")
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """

st.markdown(hide_default_format, unsafe_allow_html=True)
st.title("Predicting listings prices")


@st.cache_data
def load_data():
    testing = pd.read_csv("data/testing_df.csv")
    testing.drop(columns=testing.columns[0], axis=1, inplace=True)
    testing = testing.loc[testing["price"] < 1000, :]
    train_set, test_set = train_test_split(testing, test_size=0.2, random_state=874631)
    X_train = train_set.drop(["price"], axis=1)
    X_test = test_set.drop(["price"], axis=1)
    y_train = train_set["price"]
    y_test = test_set["price"]
    return X_train, X_test, y_train, y_test


@st.cache_resource
def load_models():
    MLPR = joblib.load("pickle/MLPR_less1k.pkl")
    LinearSVR = joblib.load("pickle/LinearSVR.pkl")
    RFR = joblib.load("pickle/RFR.pkl")
    KNNR = joblib.load("pickle/KNNR.pkl")
    NuSVR = joblib.load("pickle/NuSVR.pkl")

    return {
        "Multi Layer Perceptron Regression": MLPR,
        "Linear Support Vector Regression": LinearSVR,
        "Random Forest Regression": RFR,
        "K-Nearest Neighbors Regression": KNNR,
        "Nu Support Vector Regression": NuSVR
    }


models = load_models()
X_train, X_test, y_train, y_test = load_data()


st.markdown("## Choose model and run prediction on test data")
prediction_text = """
You can choose among five different Machine Learning models to execute the regression task.
Once the model is done with the prediction, the page will show you some metrics about
the goodness of the model in predicting the AirBnB listings prices.
It will also show some visuals to better grasp the prediction capabilities of the chosen model.
"""

st.markdown(prediction_text)

selected_model = st.selectbox("Select a model:", list(models.keys()))

#progress_text = "Switching to the needed model. Please wait."
#my_bar = st.progress(0, text=progress_text)
#for percent_complete in range(100):
#    time.sleep(0.02)
#    my_bar.progress(percent_complete + 1, text=progress_text)
#time.sleep(2)
#my_bar.empty()


if st.button("Predict listings prices"):
    best_pipe = models[selected_model]
    pred = best_pipe.predict(X_test)

    st.markdown("### Model")
    st.markdown(f"You decided to predict AirBnB listings prices with a {selected_model} model.")
    st.markdown("The following is the model pipeline that is used to execute the prediction")
    st.code(best_pipe, language="python")

    st.markdown("### Model performance metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Explained variance score",
            round(explained_variance_score(y_true=y_test, y_pred=pred), 2),
            label_visibility="visible",
            help="""
            Tests how well the prediction of the model explains the variance of the actual data.
            
            Higher is better, with 1 being the best possible result.
            """
        )

    with col2:
        st.metric(
            "Mean Absolute Error (MAE)",
            f"${round(mean_absolute_error(y_true=y_test, y_pred=pred), 2)}",
            label_visibility="visible",
            help="""
            Average absolute difference between the predicted values and the actual values.
            
            Lower is better.
            """
        )

    with col3:
        st.metric(
            "Mean Absolute Percentage Error (MAPE)",
            f"{round(100 * mean_absolute_percentage_error(y_true=y_test, y_pred=pred), 2)}%",
            label_visibility="visible",
            help="""
            Average absolute difference between the predicted values and the actual values, measured as a percentage difference
            between the actual value and the actual value.
            
            Lower is better.
            """
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Median absolute error",
            f"${round(median_absolute_error(y_true=y_test, y_pred=pred), 2)}",
            label_visibility="visible",
            help=""""
            Median difference between the actual values and predicted values.
            
            Lower is better.
            """
        )

    with col5:
        st.metric(
            "Mean Squared Error (MSE)",
            round(mean_squared_error(y_true=y_test, y_pred=pred), 2),
            label_visibility="visible",
            help="""
            Average squared difference between the predicted values and the actual values.
            
            Lower is better.
            """
        )

    with col6:
        st.metric(
            "R squared",
            round(r2_score(y_true=y_test, y_pred=pred), 2),
            label_visibility="visible",
            help="""
            Proportion of variance in the dependent variable that can be explained by the independent variable.
            
            Higher is better, with 1 being the best possible result.
            """
        )

    if selected_model == "Multi Layer Perceptron Regression":
        #graph1, graph2, graph3 = st.columns(3)

        st.markdown("### Loss Curve")

        st.markdown("""
        The following visual shows how the process of loss function optimization changes throughtout time, plotted against the number of training epochs.
        Loss is a measure of how well the model's predictions match the actual target values and it is calculated using a loss function.
        """)
        #with graph1:
        plt.figure(figsize=(5, 4))
        plt.plot(best_pipe["Model"].loss_curve_, label='Loss Curve', color='blue')  # Adjust based on your model
        plt.title('Loss Curve during Training')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        st.markdown("### True vs Predicted values")

        st.markdown("""
        The following graph shows the relationship between the True Values and the Predicted Values.
        The best possible result is a point numb that is overlapping with the 45° dashed line.
        """)

        #with graph2:
        plt.figure(figsize=(5, 4))
        plt.scatter(y_test, pred, color='orange', label='Predictions')
        plt.plot([min(y_test), max(y_test)], [min(pred), max(pred)], 'k--', lw=2, label='Perfect Prediction')
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        st.markdown("### Prediction Error")
        st.markdown("""
        The following graph shows the relationship between the values predicted by the model and the Residuals.
        The more the points are evenly spread throughout the space, the more it indicates that there is no relationship
        between the predicted value and the errors in executing the prediction.
         
        This graph is used to observe how much variance is in the model.
        """)
        #with graph3:
        display = PredictionErrorDisplay(y_true=y_test, y_pred=pred)
        display.plot()
        st.pyplot(plt)

    else:
        #graph1, graph2 = st.columns(2)

        st.markdown("### True vs Predicted values")
        st.markdown("""
        The following graph shows the relationship between the True Values and the Predicted Values.
        The best possible result is a point numb that is overlapping with the 45° dashed line.
        """)
        #with graph1:
        plt.figure(figsize=(5, 4))
        plt.scatter(y_test, pred, color='orange', label='Predictions')
        plt.plot([min(y_test), max(y_test)], [min(pred), max(pred)], 'k--', lw=2, label='Perfect Prediction')
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        st.markdown("### Prediction Error")
        st.markdown("""
        The following graph shows the relationship between the values predicted by the model and the Residuals.
        The more the points are evenly spread throughout the space, the more it indicates that there is no relationship
        between the predicted value and the errors in executing the prediction.

        This graph is used in order to observe how much variance is in the model.
        """)
        #with graph2:
        display = PredictionErrorDisplay(y_true=y_test, y_pred=pred)
        display.plot()
        st.pyplot(plt)
