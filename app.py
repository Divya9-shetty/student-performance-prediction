
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_performance_model.joblib"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_bundle()

st.title("🎓 Student Performance Prediction")
st.write(
    "Predict a student's estimated final grade (G3, out of 20) "
    "using selected educational, family, and lifestyle characteristics."
)

if bundle is None:
    st.error("The trained model file was not found.")
    st.stop()

st.caption(f"Model: {bundle['model_name']}")

feature_columns = bundle["feature_columns"]
categorical_cols = bundle["categorical_cols"]
numerical_cols = bundle["numerical_cols"]

# Only these features will be displayed in the app.
display_features = [
    "studytime", "failures", "absences", "higher",
    "Medu", "Fedu", "schoolsup", "famsup",
    "age", "health", "internet", "traveltime",
    "activities", "goout", "Dalc", "Walc"
]

# Keep only features that actually exist in the saved model.
display_features = [
    feature for feature in display_features
    if feature in feature_columns
]

# Features not displayed will use training-data defaults.
hidden_features = [
    feature for feature in feature_columns
    if feature not in display_features
]

with st.form("student_details"):
    st.subheader("Enter student details")
    input_data = {}
    left, right = st.columns(2)

    for i, feature in enumerate(display_features):
        area = left if i % 2 == 0 else right
        label = feature.replace("_", " ").title()

        # Friendlier labels
        custom_labels = {
            "studytime": "Weekly Study Time",
            "failures": "Previous Class Failures",
            "absences": "Number of Absences",
            "higher": "Wants Higher Education",
            "Medu": "Mother's Education",
            "Fedu": "Father's Education",
            "schoolsup": "School Educational Support",
            "famsup": "Family Educational Support",
            "age": "Student Age",
            "health": "Current Health (1–5)",
            "internet": "Internet Access at Home",
            "traveltime": "Travel Time to School",
            "activities": "Extracurricular Activities",
            "goout": "Going Out with Friends",
            "Dalc": "Workday Alcohol Consumption",
            "Walc": "Weekend Alcohol Consumption",
        }
        label = custom_labels.get(feature, label)

        with area:
            if feature in categorical_cols:
                options = bundle["categorical_options"].get(feature, [])
                default = bundle["categorical_defaults"].get(feature)

                if not options:
                    st.warning(f"No known values available for {label}.")
                    continue

                default_index = options.index(default) if default in options else 0

                input_data[feature] = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    key=f"cat_{feature}"
                )

            elif feature in numerical_cols:
                info = bundle["numerical_defaults"][feature]
                minimum = info["min"]
                maximum = info["max"]
                default = min(max(info["median"], minimum), maximum)
                step = 1.0 if info["integer"] else 0.1

                input_data[feature] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    key=f"num_{feature}"
                )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary"
    )

if submitted:
    try:
        # Fill hidden features with defaults from the training data.
        row = {}

        for feature in feature_columns:
            if feature in input_data:
                row[feature] = input_data[feature]

            elif feature in categorical_cols:
                row[feature] = bundle["categorical_defaults"][feature]

            elif feature in numerical_cols:
                row[feature] = bundle["numerical_defaults"][feature]["median"]

        input_df = pd.DataFrame([row], columns=feature_columns)
        prediction = float(bundle["model"].predict(input_df)[0])

        low = bundle.get("target_min", 0.0)
        high = bundle.get("target_max", 20.0)
        prediction_display = min(max(prediction, low), high)

        st.subheader("Prediction")
        st.metric(
            "Estimated Final Grade (G3)",
            f"{prediction_display:.2f} / 20"
        )

        if prediction_display >= 16:
            st.success("Estimated category: High")
        elif prediction_display >= 10:
            st.info("Estimated category: Moderate")
        else:
            st.warning("Estimated category: Low")

        st.caption(
            "This is a machine-learning estimate, not a guaranteed grade "
            "or a formal assessment of a student. Hidden features use "
            "default values, which may affect the prediction."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
