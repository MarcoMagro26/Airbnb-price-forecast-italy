import streamlit as st


st.set_page_config(
    page_title="Introduction"
)

st.write("# Welcome to the Italian AirBnB pricing App!")

st.sidebar.success("Select a page from the list above.")

st.markdown(
    """
    ## Objective
    
    The primary objective of this project is to forecast Airbnb listings prices across various locations in Italy.
    More precisely:
    
    - Rome
    - Milan
    - Venice
    - Florence
    - Bologna
    - Naples
    - Bergamo
    
    By leveraging different modeling techniques, we aim to provide a tool to observe the performance of various prediction
    models to accomplish the regression task at hand.

    ## Data Source
    
    Data for this project has been sourced from various datasets extracted from the "Inside Airbnb" platform.
    Each dataset includes thousands of listings, with each listing containing essential features such as price, location,
    amenities, host information, etc,which are crucial for building predictive models.

    ## Methodology

    To achieve the project's goal, we merged the various dataset, in order to enrich the pool of data and increase both the
    training and the test set size.
    Several machine learning regression models are employed and evaluated, each aiming to capture the characteristics of the data.

    ## Model Evaluation

    The effectiveness of each model has been assessed some performance metrics, including:
    
    - explained variance score, which measures the proportion of variance in the target variable that can be explained by the model;
    - Mean Absolute Error (MAE), which provides an average of the absolute differences between predicted and actual prices;
    - Mean Absolute Percentage Error (MAPE), expressing prediction accuracy as a percentage, facilitating comparison across different scales;
    - Mean Squared Error (MSE), which calculates the average of the squares of the errors, emphasizing larger differences in predictions;
    
    The project goal is to deliver a tool to observe forecast capabilities of different models and understand which is
    better suited to accomplish the task in a more precise manner.
"""
)

st.markdown(
    """
    ## Next steps
    - Check out [the project repository](https://github.com/FiliTol/airbnb-price-forecasting) to analyze the technical details involving the implementation;
    - Jump into our [slides presentation]();
    - Are you wondering how we did this website? Take a look [here](https://docs.streamlit.io) and deep into the Streamlit platform.
"""
)

