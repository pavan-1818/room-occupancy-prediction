import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Room Occupancy Detection",
    page_icon="🏠",
    layout="wide"
)


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    logistic_model = joblib.load(
        "logistic_regression_model.pkl"
    )

    random_forest_model = joblib.load(
        "random_forest_model.pkl"
    )

    svm_model = joblib.load(
        "svm_model.pkl"
    )

    feature_columns = joblib.load(
        "feature_columns.pkl"
    )

    return (
        logistic_model,
        random_forest_model,
        svm_model,
        feature_columns
    )


(
    logistic_model,
    random_forest_model,
    svm_model,
    feature_columns
) = load_models()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Prediction",
        "Model Comparison"
    ]
)


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.title(
        "🏠 Room Occupancy Detection"
    )

    st.subheader(
        "Environmental Sensor Based Machine Learning System"
    )

    st.write(
        """
        This project predicts the number of people present
        inside a room using environmental sensor readings.
        """
    )

    st.divider()

    # ------------------------------------------------------
    # PROJECT DESCRIPTION
    # ------------------------------------------------------

    st.header("📌 About the Project")

    st.write(
        """
        Environmental sensors such as temperature, light,
        sound, CO₂ and PIR motion sensors can provide useful
        information about the occupancy of a room.

        Machine learning algorithms are used to learn the
        relationship between these sensor measurements and
        the number of people present in the room.
        """
    )

    # ------------------------------------------------------
    # MODELS
    # ------------------------------------------------------

    st.header("🤖 Machine Learning Models")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "Logistic Regression"
        )

        st.write(
            """
            A classification algorithm that estimates the
            probability of different occupancy classes.
            """
        )

    with col2:

        st.subheader(
            "Random Forest"
        )

        st.write(
            """
            An ensemble learning algorithm that combines
            multiple decision trees.
            """
        )

    with col3:

        st.subheader(
            "SVM"
        )

        st.write(
            """
            Support Vector Machine finds a decision boundary
            that separates different occupancy classes.
            """
        )

    st.divider()

    # ------------------------------------------------------
    # SENSOR TYPES
    # ------------------------------------------------------

    st.header("📡 Sensors Used")

    sensor_data = {
        "Sensor": [
            "Temperature",
            "Light",
            "Sound",
            "CO₂",
            "PIR Motion"
        ],

        "Purpose": [
            "Measures room temperature",
            "Measures light intensity",
            "Measures sound level",
            "Measures CO₂ concentration",
            "Detects human movement"
        ]
    }

    st.dataframe(
        pd.DataFrame(sensor_data),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.info(
        """
        Use the Prediction page to enter sensor values
        and predict room occupancy.
        """
    )


# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Prediction":

    st.title(
        "🔮 Room Occupancy Prediction"
    )

    st.write(
        "Enter the environmental sensor measurements below."
    )

    st.divider()

    # ------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------

    st.header("🌡️ Temperature Sensors")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        S1_Temp = st.number_input(
            "S1 Temperature",
            value=25.0,
            step=0.1
        )

    with col2:
        S2_Temp = st.number_input(
            "S2 Temperature",
            value=25.0,
            step=0.1
        )

    with col3:
        S3_Temp = st.number_input(
            "S3 Temperature",
            value=25.0,
            step=0.1
        )

    with col4:
        S4_Temp = st.number_input(
            "S4 Temperature",
            value=25.0,
            step=0.1
        )


    # ------------------------------------------------------
    # LIGHT
    # ------------------------------------------------------

    st.header("💡 Light Sensors")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        S1_Light = st.number_input(
            "S1 Light",
            value=300.0,
            step=1.0
        )

    with col2:
        S2_Light = st.number_input(
            "S2 Light",
            value=300.0,
            step=1.0
        )

    with col3:
        S3_Light = st.number_input(
            "S3 Light",
            value=300.0,
            step=1.0
        )

    with col4:
        S4_Light = st.number_input(
            "S4 Light",
            value=300.0,
            step=1.0
        )


    # ------------------------------------------------------
    # SOUND
    # ------------------------------------------------------

    st.header("🔊 Sound Sensors")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        S1_Sound = st.number_input(
            "S1 Sound",
            value=0.10,
            step=0.01
        )

    with col2:
        S2_Sound = st.number_input(
            "S2 Sound",
            value=0.10,
            step=0.01
        )

    with col3:
        S3_Sound = st.number_input(
            "S3 Sound",
            value=0.10,
            step=0.01
        )

    with col4:
        S4_Sound = st.number_input(
            "S4 Sound",
            value=0.10,
            step=0.01
        )


    # ------------------------------------------------------
    # CO2
    # ------------------------------------------------------

    st.header("🫧 CO₂ Sensor")

    col1, col2 = st.columns(2)

    with col1:

        S5_CO2 = st.number_input(
            "CO₂ Concentration",
            value=600.0,
            step=10.0
        )

    with col2:

        S5_CO2_Slope = st.number_input(
            "CO₂ Slope",
            value=0.0,
            step=0.1
        )


    # ------------------------------------------------------
    # PIR
    # ------------------------------------------------------

    st.header("🚶 PIR Motion Sensors")

    col1, col2 = st.columns(2)

    with col1:

        S6_PIR = st.selectbox(
            "S6 PIR",
            [0, 1],
            format_func=lambda x:
                "No Motion" if x == 0 else "Motion Detected"
        )

    with col2:

        S7_PIR = st.selectbox(
            "S7 PIR",
            [0, 1],
            format_func=lambda x:
                "No Motion" if x == 0 else "Motion Detected"
        )


    # ------------------------------------------------------
    # DATE
    # ------------------------------------------------------

    st.header("📅 Date and Time")

    col1, col2, col3 = st.columns(3)

    with col1:

        Year = st.number_input(
            "Year",
            value=2015,
            step=1
        )

    with col2:

        Month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=2,
            step=1
        )

    with col3:

        Day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=5,
            step=1
        )


    # ------------------------------------------------------
    # TIME
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        DayOfWeek = st.number_input(
            "Day of Week",
            min_value=0,
            max_value=6,
            value=3,
            step=1
        )

    with col2:

        Hour = st.number_input(
            "Hour",
            min_value=0,
            max_value=23,
            value=14,
            step=1
        )

    with col3:

        Minute = st.number_input(
            "Minute",
            min_value=0,
            max_value=59,
            value=30,
            step=1
        )

    with col4:

        Second = st.number_input(
            "Second",
            min_value=0,
            max_value=59,
            value=0,
            step=1
        )


    # ======================================================
    # PREDICT BUTTON
    # ======================================================

    st.divider()

    predict_button = st.button(
        "🔍 PREDICT ROOM OCCUPANCY",
        type="primary",
        use_container_width=True
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    if predict_button:

        # ----------------------------------------------
        # Create input dataframe
        # ----------------------------------------------

        input_data = pd.DataFrame([{

            "S1_Temp": S1_Temp,
            "S2_Temp": S2_Temp,
            "S3_Temp": S3_Temp,
            "S4_Temp": S4_Temp,

            "S1_Light": S1_Light,
            "S2_Light": S2_Light,
            "S3_Light": S3_Light,
            "S4_Light": S4_Light,

            "S1_Sound": S1_Sound,
            "S2_Sound": S2_Sound,
            "S3_Sound": S3_Sound,
            "S4_Sound": S4_Sound,

            "S5_CO2": S5_CO2,
            "S5_CO2_Slope": S5_CO2_Slope,

            "S6_PIR": S6_PIR,
            "S7_PIR": S7_PIR,

            "Year": Year,
            "Month": Month,
            "Day": Day,
            "DayOfWeek": DayOfWeek,

            "Hour": Hour,
            "Minute": Minute,
            "Second": Second

        }])


        # ----------------------------------------------
        # Ensure correct column order
        # ----------------------------------------------

        input_data = input_data[
            feature_columns
        ]


        # ----------------------------------------------
        # Make predictions
        # ----------------------------------------------

        logistic_prediction = logistic_model.predict(
            input_data
        )[0]

        rf_prediction = random_forest_model.predict(
            input_data
        )[0]

        svm_prediction = svm_model.predict(
            input_data
        )[0]


        # ==================================================
        # RESULTS
        # ==================================================

        st.divider()

        st.header(
            "📊 Model Predictions"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            st.subheader(
                "Logistic Regression"
            )

            st.metric(
                "Occupancy",
                f"{logistic_prediction} Person(s)"
            )


        with col2:

            st.subheader(
                "Random Forest"
            )

            st.metric(
                "Occupancy",
                f"{rf_prediction} Person(s)"
            )


        with col3:

            st.subheader(
                "SVM"
            )

            st.metric(
                "Occupancy",
                f"{svm_prediction} Person(s)"
            )


        # ==================================================
        # FINAL PREDICTION
        # ==================================================

        predictions = [
            int(logistic_prediction),
            int(rf_prediction),
            int(svm_prediction)
        ]

        final_prediction = max(
            set(predictions),
            key=predictions.count
        )


        st.divider()

        st.header(
            "🏆 Final Prediction"
        )


        if final_prediction == 0:

            st.success(
                "🟢 ROOM IS EMPTY — 0 PEOPLE"
            )

        elif final_prediction == 1:

            st.info(
                "🔵 ROOM OCCUPANCY — 1 PERSON"
            )

        elif final_prediction == 2:

            st.warning(
                "🟡 ROOM OCCUPANCY — 2 PEOPLE"
            )

        elif final_prediction == 3:

            st.error(
                "🔴 ROOM OCCUPANCY — 3 PEOPLE"
            )


        # ==================================================
        # MODEL AGREEMENT
        # ==================================================

        if (
            logistic_prediction ==
            rf_prediction ==
            svm_prediction
        ):

            st.success(
                "✅ All three models agree on the prediction!"
            )

        else:

            st.warning(
                "⚠️ The models produced different predictions."
            )


        # ==================================================
        # INPUT DATA
        # ==================================================

        with st.expander(
            "🔎 View Sensor Input"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


# ==========================================================
# MODEL COMPARISON PAGE
# ==========================================================

elif page == "Model Comparison":

    st.title(
        "📊 Model Comparison"
    )

    st.write(
        """
        Comparison of the three machine learning algorithms
        used for room occupancy detection.
        """
    )

    st.divider()


    # ------------------------------------------------------
    # LOAD RESULTS
    # ------------------------------------------------------

    try:

        results_df = pd.read_csv(
            "model_results.csv"
        )

        st.subheader(
            "Model Performance"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # Accuracy chart
        # --------------------------------------------------

        st.subheader(
            "Accuracy Comparison"
        )

        chart_data = results_df.set_index(
            "Model"
        )["Accuracy"]

        st.bar_chart(
            chart_data
        )


    except FileNotFoundError:

        st.warning(
            "model_results.csv was not found."
        )


    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    try:

        feature_importance = pd.read_csv(
            "feature_importance.csv"
        )

        st.divider()

        st.subheader(
            "🌳 Random Forest Feature Importance"
        )

        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )

    except FileNotFoundError:

        st.info(
            "feature_importance.csv was not found."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Room Occupancy Detection using Machine Learning"
)

st.sidebar.caption(
    "Logistic Regression • Random Forest • SVM"
)