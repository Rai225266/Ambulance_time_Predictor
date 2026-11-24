import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

model = joblib.load("ambulance_chatbot_model.pkl")

st.set_page_config(
    page_title="Ambulance AI | Smart Dispatch",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #101820;
    color: #f1f5f9;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1500px;
}

h1 {
    color: #ffb52e !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #f1f5f9 !important;
}

p, label {
    color: #d5e0e7 !important;
}

.panel {
    background: linear-gradient(145deg,#1c303d,#14232d);
    border: 1px solid #3b5665;
    border-radius: 14px;
    padding: 22px;
}

.section-title {
    color: #ffb52e;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

.metric-card {
    background: #101c25;
    border: 1px solid #3d5360;
    border-radius: 12px;
    padding: 18px;
    min-height: 135px;
}

.metric-label {
    color: #aabac4;
    font-size: 12px;
    font-weight: bold;
}

.metric-value {
    color: #ffbd35;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

.small-text {
    color: #9babb5;
    font-size: 12px;
    margin-top: 8px;
}

div.stButton > button {
    background: linear-gradient(90deg,#ffae00,#ffc72c);
    color: #17212a;
    border: none;
    border-radius: 8px;
    height: 50px;
    width: 100%;
    font-weight: 800;
    font-size: 15px;
}

div.stButton > button:hover {
    background: #ffd15c;
    color: #111;
}

hr {
    border-color: #344b58;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# CATEGORY MAPPINGS
# ==========================================
# PROVISIONAL MAPPINGS:
# Verify category meanings against original dataset
# before treating these as real-world labels.

traffic_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

weather_map = {
    "Fair": 0,
    "Cloudy": 0,
    "Rainy": 1,
    "Thunderstorm": 1,
    "Foggy": 2
}

road_map = {
    "Good": 0,
    "Fair": 1,
    "Poor": 2,
    "Bad": 2
}

time_map = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3
}


# ==========================================
# HEADER
# ==========================================

st.title(" AMBULANCE TIME PREDICTOR")

st.markdown("""
<p style="color:#9aadb9; font-size:13px; letter-spacing:1px;">
INTELLIGENT EMERGENCY DISPATCH & DELAY RISK ANALYSIS
</p>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# MAIN DASHBOARD
# ==========================================

left, right = st.columns([1.05, 1], gap="large")


# ==========================================
# LEFT PANEL: INCIDENT INPUT
# ==========================================

with left:

    st.markdown(
        '<div class="panel"><div class="section-title">'
        '01 &nbsp; INCIDENT INPUT</div></div>',
        unsafe_allow_html=True
    )

    location = st.text_input(
        "📍 EMERGENCY LOCATION",
        placeholder="Enter location (e.g. Solan)"
    )

    distance = st.slider(
        "DISTANCE TO SCENE",
        min_value=0.1,
        max_value=100.0,
        value=8.2,
        step=0.1,
        format="%.1f km"
    )

    col1, col2 = st.columns(2)

    with col1:

        traffic = st.selectbox(
            "🚦 TRAFFIC",
            ["Low", "Moderate", "High"]
        )

        road = st.selectbox(
            "🛣️ ROAD CONDITION",
            ["Good", "Fair", "Poor", "Bad"]
        )

    with col2:

        weather = st.selectbox(
            "🌦️ WEATHER",
            [
                "Fair",
                "Cloudy",
                "Rainy",
                "Thunderstorm",
                "Foggy"
            ]
        )

        time = st.selectbox(
            "🕒 TIME",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )

    st.write("")

    predict = st.button(
        "▶ RUN PREDICTION",
        use_container_width=True
    )


# ==========================================
# RIGHT PANEL: AI PREDICTION
# ==========================================

with right:

    st.markdown(
        '<div class="panel"><div class="section-title">'
        '02 &nbsp; AI PREDICTION</div></div>',
        unsafe_allow_html=True
    )

    if predict:

        if not location.strip():

            st.warning("Please enter emergency location.")

        else:

            # ----------------------------------
            # PREPARE MODEL INPUT
            # ----------------------------------

            input_data = pd.DataFrame([{
                "Distance_km": distance,
                "Traffic_Level": traffic_map[traffic],
                "Weather": weather_map[weather],
                "Road_Type": road_map[road],
                "Time_of_Day": time_map[time]
            }])

            # ----------------------------------
            # ML MODEL PREDICTION
            # ----------------------------------

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            delayed_probability = probabilities[
                classes.index(0)
            ]

            on_time_probability = probabilities[
                classes.index(1)
            ]

            risk_percent = delayed_probability * 100

            if prediction == 1:
                status = "ON TIME"
                status_color = "#42d99b"
            else:
                status = "DELAYED"
                status_color = "#ff625c"

            if risk_percent < 30:
                risk = "LOW"
                risk_color = "#42d99b"

            elif risk_percent < 60:
                risk = "MEDIUM"
                risk_color = "#ffbd35"

            else:
                risk = "HIGH"
                risk_color = "#ff625c"


            # ==================================
            # RESPONSE TIME ESTIMATION
            # ==================================
            # This is a rule-based estimate,
            # NOT a trained regression prediction.

            traffic_speed = {
                "Low": 40,
                "Moderate": 30,
                "High": 20
            }

            weather_penalty = {
                "Fair": 0,
                "Cloudy": 2,
                "Rainy": 5,
                "Thunderstorm": 8,
                "Foggy": 6
            }

            road_penalty = {
                "Good": 0,
                "Fair": 3,
                "Poor": 6,
                "Bad": 8
            }

            time_penalty = {
                "Morning": 2,
                "Afternoon": 0,
                "Evening": 4,
                "Night": 1
            }

            effective_speed = (
                traffic_speed[traffic]
                - weather_penalty[weather]
                - road_penalty[road]
                - time_penalty[time]
            )

            effective_speed = max(effective_speed, 8)

            # Estimated arrival time
            response_time = (
                distance / effective_speed
            ) * 60 + 3

            # Ideal travel time
            ideal_time = (
                distance / 40
            ) * 60 + 3

            expected_delay = max(
                0,
                response_time - ideal_time
            )


            # ==================================
            # RESULT CARDS
            # ==================================

            st.markdown("### PREDICTION OVERVIEW")

            card1, card2 = st.columns(2)

            with card1:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">
                        PREDICTED RESPONSE TIME
                    </div>
                    <div class="metric-value">
                        {response_time:.1f} min
                    </div>
                    <div class="small-text">
                        Rule-based estimated arrival time
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with card2:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">
                        EXPECTED DELAY
                    </div>
                    <div class="metric-value">
                        +{expected_delay:.1f} min
                    </div>
                    <div class="small-text">
                        Compared with ideal travel conditions
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")


            # ==================================
            # STATUS
            # ==================================

            st.markdown("### RESPONSE-TIME CLASSIFICATION")

            st.markdown(
                f"""
                <div style="
                    background:#172630;
                    border-left:5px solid {status_color};
                    padding:15px;
                    border-radius:8px;
                    color:{status_color};
                    font-size:22px;
                    font-weight:bold;">
                    {status}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")


            # ==================================
            # DELAY RISK
            # ==================================

            st.markdown("### DELAY RISK")

            st.progress(float(delayed_probability))

            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between">
                    <b style="color:{risk_color}">
                        {risk_percent:.1f}%
                    </b>
                    <b style="color:{risk_color}">
                        {risk} RISK
                    </b>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")


            # ==================================
            # MODEL PROBABILITIES
            # ==================================

            st.markdown("### MODEL PREDICTION")

            st.write(
                f"🟢 On-Time Probability: "
                f"**{on_time_probability * 100:.1f}%**"
            )

            st.write(
                f"🔴 Delayed Probability: "
                f"**{delayed_probability * 100:.1f}%**"
            )

            st.divider()


            # ==================================
            # INCIDENT SUMMARY
            # ==================================

            st.markdown("### INCIDENT SUMMARY")

            st.write(f"📍 Location: {location}")
            st.write(f"📏 Distance: {distance} km")
            st.write(f"🚦 Traffic: {traffic}")
            st.write(f"🌦️ Weather: {weather}")
            st.write(f"🛣️ Road Condition: {road}")
            st.write(f"🕒 Time: {time}")

    else:

        st.info(
            "Enter incident details and click RUN PREDICTION."
        )

        st.markdown("### Awaiting emergency information...")

        st.caption(
            "AI prediction results will appear here."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#8497a3;
    font-size:12px;
    padding:10px;">

    AMBULANCE AI RESPONSE PREDICTOR<br>
    LOGISTIC REGRESSION CLASSIFICATION MODEL<br><br>

    Academic ML Prototype | Not for real emergency dispatch

</div>
""", unsafe_allow_html=True)

st.caption(
    "Response time and expected delay are rule-based estimates. "
    "Delay risk and status are generated by the trained classification model. "
    "Category mappings are provisional and require dataset verification."
)
