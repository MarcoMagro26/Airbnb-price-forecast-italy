import joblib
import random
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Custom Data Entry")

st.title("Custom Values Entry for Airbnb Data")

st.markdown("""
Select the model you're willing to use to execute the prediction.
You can decide to draw some random listings from the test set and predict their price.
If willing to, you can manually fill in the details for a custom listing and predict its price with the choosen model. 
""")

sampled_observation = None


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
    return train_set


df = load_data()
models = load_models()

tab1, tab2 = st.tabs(["Sample data", "Insert manually"])

if 'actual_price' not in st.session_state:
    st.session_state['actual_price'] = []
if 'predicted_price' not in st.session_state:
    st.session_state['predicted_price'] = []
if 'point_color' not in st.session_state:
    st.session_state['point_color'] = []
#st.write(st.session_state)

with tab1:
    selected_model = st.selectbox("Select a model:", list(models.keys()), key='unique_key_1')
    st.markdown("""
    You can submit multiple observations by pressing repeatedly the submission button below.
    The predicted price will be drawn in the *True vs Predicted* graph, with a different color based on the 
    model chosen for the single prediction.
    """)
    if st.button("Sample random observation from test set and submit it to the prediction model",
                 help="This will override any custom listing property provided in the form"):
        sampled_observation = df.sample(n=1, axis=0)
        actual_price = sampled_observation["price"].iloc[0]
        sampled_observation = sampled_observation.drop(["price"], axis=1)
        st.success("A random observation has been loaded and submitted")
        st.json(sampled_observation.to_dict(orient="records")[0], expanded=False)

        best_pipe = models[selected_model]
        pred = best_pipe.predict(sampled_observation)

        actual, predicted = st.columns(2)

        with actual:
            st.metric(
                "Actual price",
                f"${round(actual_price, 2)}",
                label_visibility="visible",
                help="""
                Actual price of the listing retrieved from the initial dataset.
                """
            )

        with predicted:
            st.metric(
                "Predicted price",
                f"${round(pred[0], 2)}",
                label_visibility="visible",
                help="""
                Predicted price of the listing that was randomly sampled
                """
            )

        st.session_state['actual_price'].append(actual_price)
        st.session_state['predicted_price'].append(pred[0])

        st.markdown("### Scatter Plot of Actual vs Predicted Prices")
        fig, ax = plt.subplots()

        point_color = None
        if selected_model == "Multi Layer Perceptron Regression":
            point_color = 'blue'
        elif selected_model == "Linear Support Vector Regression":
            point_color = 'red'
        elif selected_model == "Random Forest Regression":
            point_color = 'green'
        elif selected_model == "K-Nearest Neighbors Regression":
            point_color = 'yellow'
        else:
            point_color = 'black'

        st.session_state['point_color'].append(point_color)

        ax.scatter(
            st.session_state["actual_price"],
            st.session_state["predicted_price"],
            c=st.session_state['point_color'],
            alpha=0.7
        )

        price_range = max(
            st.session_state["actual_price"] + st.session_state["predicted_price"]
        )
        ax.plot([0, price_range], [0, price_range], color='gray', linestyle='--', label="Perfect prediction line")

        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='', markersize=10, label="MLPR"),
            plt.Line2D([0], [0], marker='o', color='red', linestyle='', markersize=10, label="LinearSVR"),
            plt.Line2D([0], [0], marker='o', color='green', linestyle='', markersize=10, label="RFR"),
            plt.Line2D([0], [0], marker='o', color='yellow', linestyle='', markersize=10, label="KNNR"),
            plt.Line2D([0], [0], marker='o', color='black', linestyle='', markersize=10, label="NuSVR"),
            plt.Line2D([0], [0], color='gray', linestyle='--', label="Perfect prediction")
        ]

        ax.legend(handles=handles, loc="best", title="Models")

        ax.set_title("Actual vs Predicted Prices")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)
        fig, ax = plt.subplots()

with tab2:
    selected_model = st.selectbox("Select a model:", list(models.keys()), key='unique_key_2')
    with st.form("input_form"):
        st.subheader("General features")
        #host_id = st.text_input("Host ID")
        host_response_rate = st.number_input("Host Response Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
        host_acceptance_rate = st.number_input("Host Acceptance Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
        #price = st.number_input("Price (USD)", min_value=0.0, step=0.01)
        host_since = st.date_input("Host Since (date)")
        first_review = st.date_input("First Review (date)")
        last_review = st.date_input("Last Review (date)")
        host_location = st.text_input("Host Location")
        host_response_time = st.selectbox("Host Response Time",
                                          ["Within an hour", "Within a few hours", "Within a day", "Not available"])
        host_is_superhost = st.checkbox("Is Superhost")
        host_listings_count = st.number_input("Number of Listings by Host", min_value=0)
        host_has_profile_pic = st.checkbox("Host Has Profile Picture")
        host_identity_verified = st.checkbox("Host Identity Verified")
        latitude = st.number_input("Latitude", format="%.6f")
        longitude = st.number_input("Longitude", format="%.6f")
        property_type = st.text_input("Property Type")
        room_type = st.text_input("Room Type")
        accommodates = st.number_input("Accommodates", min_value=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)
        bathrooms_text = st.text_input("Bathrooms Description")
        minimum_nights = st.number_input("Minimum Nights", min_value=1)
        maximum_nights = st.number_input("Maximum Nights", min_value=1)
        number_of_reviews = st.number_input("Number of Reviews", min_value=0)
        review_scores_rating = st.number_input("Review Scores Rating", min_value=0.0, max_value=100.0, step=0.1)
        review_scores_checkin = st.number_input("Review Scores Checkin", min_value=0.0, max_value=10.0, step=0.1)
        review_scores_location = st.number_input("Review Scores Location", min_value=0.0, max_value=10.0, step=0.1)
        review_scores_value = st.number_input("Review Scores Value", min_value=0.0, max_value=10.0, step=0.1)
        reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, step=0.1)
        df_city_location = st.text_input("City Location")
        listing_city_pop = st.number_input("City Population", min_value=0)
        email_verification = st.checkbox("Email Verification")
        phone_verification = st.checkbox("Phone Verification")
        work_email_verification = st.checkbox("Work Email Verification")

        # Amenities
        st.subheader("Amenities")
        amenities_internet = st.checkbox("Internet")
        amenities_self_checkin = st.checkbox("Self-Checkin")
        amenities_host_greeting = st.checkbox("Host Greeting")
        amenities_pool = st.checkbox("Pool")
        amenities_oven = st.checkbox("Oven")
        amenities_microwave = st.checkbox("Microwave")
        amenities_garden = st.checkbox("Garden")
        amenities_streaming = st.checkbox("Streaming Services")
        amenities_gym = st.checkbox("Gym")
        amenities_elevator = st.checkbox("Elevator")
        amenities_heating = st.checkbox("Heating")
        amenities_air_conditioning = st.checkbox("Air Conditioning")
        amenities_workspace = st.checkbox("Workspace")
        amenities_freezer = st.checkbox("Freezer")
        amenities_first_aid_kit = st.checkbox("First Aid Kit")
        amenities_dishwasher = st.checkbox("Dishwasher")
        amenities_long_term_stays = st.checkbox("Long-term Stays")
        amenities_pets_allowed = st.checkbox("Pets Allowed")
        amenities_bathtube = st.checkbox("Bathtub")
        amenities_bbq_grill = st.checkbox("BBQ Grill")
        amenities_lake_bay_view = st.checkbox("Lake/Bay View")

        # Description analysis
        st.subheader("Description Analysis")
        description_word_count = st.number_input("Description Word Count", min_value=0)
        description_sentiment_polarity = st.number_input("Description Sentiment Polarity", min_value=-1.0,
                                                         max_value=1.0, step=0.01)
        description_sentiment_subjectivity = st.number_input("Description Sentiment Subjectivity", min_value=0.0,
                                                             max_value=1.0, step=0.01)

        # Beds and scraping date
        beds_for_bedroom = st.number_input("Beds per Bedroom", min_value=0)
        scraping_date = st.date_input("Scraping Date (date)")

        # Submit button
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.success("Custom values successfully submitted!")
        user_data = {
            #"host_id": host_id,
            "host_response_rate": host_response_rate,
            "host_acceptance_rate": host_acceptance_rate,
            #"price": price,
            "host_since": str(host_since),
            "first_review": str(first_review),
            "last_review": str(last_review),
            "host_location": host_location,
            "host_response_time": host_response_time,
            "host_is_superhost": host_is_superhost,
            "host_listings_count": host_listings_count,
            "host_has_profile_pic": host_has_profile_pic,
            "host_identity_verified": host_identity_verified,
            "latitude": latitude,
            "longitude": longitude,
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bathrooms_text": bathrooms_text,
            "minimum_nights": minimum_nights,
            "maximum_nights": maximum_nights,
            "number_of_reviews": number_of_reviews,
            "review_scores_rating": review_scores_rating,
            "review_scores_checkin": review_scores_checkin,
            "review_scores_location": review_scores_location,
            "review_scores_value": review_scores_value,
            "reviews_per_month": reviews_per_month,
            "df_city_location": df_city_location,
            "listing_city_pop": listing_city_pop,
            "email_verification": email_verification,
            "phone_verification": phone_verification,
            "work_email_verification": work_email_verification,
            "amenities": {
                "internet": amenities_internet,
                "self_checkin": amenities_self_checkin,
                "host_greeting": amenities_host_greeting,
                "pool": amenities_pool,
                "oven": amenities_oven,
                "microwave": amenities_microwave,
                "garden": amenities_garden,
                "streaming": amenities_streaming,
                "gym": amenities_gym,
                "elevator": amenities_elevator,
                "heating": amenities_heating,
                "air_conditioning": amenities_air_conditioning,
                "workspace": amenities_workspace,
                "freezer": amenities_freezer,
                "first_aid_kit": amenities_first_aid_kit,
                "dishwasher": amenities_dishwasher,
                "long_term_stays": amenities_long_term_stays,
                "pets_allowed": amenities_pets_allowed,
                "bathtube": amenities_bathtube,
                "bbq_grill": amenities_bbq_grill,
                "lake_bay_view": amenities_lake_bay_view
            },
            "description_word_count": description_word_count,
            "description_sentiment_polarity": description_sentiment_polarity,
            "description_sentiment_subjectivity": description_sentiment_subjectivity,
            "beds_for_bedroom": beds_for_bedroom,
            "scraping_date": str(scraping_date),
        }
        st.json(user_data, expanded=False)

        models = load_models()

        selected_model = st.selectbox("Select a model:", list(models.keys()))

        if st.button("Predict Price"):
            prediction = models[selected_model].predict(sampled_observation)
            st.success(f"The predicted price is: ${prediction:.2f}")

