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
    "using demographic, family, social, and educational characteristics."
)

if bundle is None:
    st.error("The trained model file was not found.")
    st.code(
        "python train_model.py\npython -m streamlit run app.py",
        language="bash"
    )
    st.stop()

st.caption(f"Model: {bundle['model_name']}")

# Original feature names required by the trained model
feature_columns = bundle["feature_columns"]
categorical_cols = bundle["categorical_cols"]
numerical_cols = bundle["numerical_cols"]

# Friendly labels shown to users.
# These labels do not change the model's original feature names.
custom_labels = {
    "school": "School",
    "sex": "Gender",
    "age": "Student Age",
    "address": "Type of Home Location",
    "famsize": "Family Size",
    "Pstatus": "Parents' Living Arrangement",
    "Medu": "Mother's Education Level",
    "Fedu": "Father's Education Level",
    "Mjob": "Mother's Occupation",
    "Fjob": "Father's Occupation",
    "reason": "Reason for Choosing the School",
    "guardian": "Student's Guardian",
    "traveltime": "Travel Time to School",
    "studytime": "Weekly Study Time",
    "failures": "Previous Class Failures",
    "schoolsup": "Extra Educational Support from School",
    "famsup": "Educational Support from Family",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery School",
    "higher": "Plans for Higher Education",
    "internet": "Internet Access at Home",
    "romantic": "In a Romantic Relationship",
    "famrel": "Family Relationship Quality",
    "freetime": "Free Time After School",
    "goout": "Frequency of Going Out with Friends",
    "Dalc": "Alcohol Consumption on Weekdays",
    "Walc": "Alcohol Consumption on Weekends",
    "health": "Current Health Status",
    "absences": "Number of School Absences",
}

# Explanations for fields with coded numeric values.
field_help = {
    "Medu": "0 = None, 1 = Primary, 2 = 5th–9th grade, "
            "3 = Secondary, 4 = Higher education.",
    "Fedu": "0 = None, 1 = Primary, 2 = 5th–9th grade, "
            "3 = Secondary, 4 = Higher education.",
    "traveltime": "1 = Less than 15 min, 2 = 15–30 min, "
                  "3 = 30–60 min, 4 = More than 60 min.",
    "studytime": "1 = Less than 2 hours, 2 = 2–5 hours, "
                 "3 = 5–10 hours, 4 = More than 10 hours per week.",
    "famrel": "1 = Very bad, 5 = Excellent.",
    "freetime": "1 = Very low, 5 = Very high.",
    "goout": "1 = Very low frequency, 5 = Very high frequency.",
    "Dalc": "1 = Very low, 5 = Very high.",
    "Walc": "1 = Very low, 5 = Very high.",
    "health": "1 = Very poor, 5 = Very good.",
    "failures": "Number of previous class failures.",
    "absences": "Number of school absences recorded.",
}

with st.form("student_details"):
    st.subheader("Enter Student Details")
    st.caption(
        "Complete the fields below. The model will use all its original "
        "input features to generate a prediction."
    )

    input_data = {}
    left, right = st.columns(2)

    # Keep ALL original model features in the form.
    for i, feature in enumerate(feature_columns):
        area = left if i % 2 == 0 else right
        label = custom_labels.get(
            feature,
            feature.replace("_", " ").title()
        )
        help_text = field_help.get(feature)

        with area:
            if feature in categorical_cols:
                options = bundle["categorical_options"].get(feature, [])
                default = bundle["categorical_defaults"].get(feature)

                if not options:
                    st.warning(
                        f"No known values available for {label}."
                    )
                    continue

                default_index = (
                    options.index(default)
                    if default in options
                    else 0
                )

                input_data[feature] = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    help=help_text,
                    key=f"cat_{feature}"
                )

            elif feature in numerical_cols:
                info = bundle["numerical_defaults"][feature]
                minimum = info["min"]
                maximum = info["max"]
                default = min(
                    max(info["median"], minimum),
                    maximum
                )
                step = 1.0 if info["integer"] else 0.1

                input_data[feature] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    help=help_text,
                    key=f"num_{feature}"
                )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary"
    )


if submitted:
    try:
        # Preserve the exact feature names and order expected by the model.
        missing_features = [
            col for col in feature_columns
            if col not in input_data
        ]

        if missing_features:
            st.error(
                "Some model inputs are missing: "
                + ", ".join(missing_features)
                + ". Please check the saved model's feature configuration."
            )
            st.stop()

        row = {
            col: input_data[col]
            for col in feature_columns
        }

        input_df = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        prediction = float(
            bundle["model"].predict(input_df)[0]
        )

        # Keep the displayed result within the dataset target range.
        low = bundle.get("target_min", 0.0)
        high = bundle.get("target_max", 20.0)

        prediction_display = min(
            max(prediction, low),
            high
        )

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
            "This is an estimate generated by a machine learning model, "
            "not a guaranteed result or a formal assessment of a student."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
