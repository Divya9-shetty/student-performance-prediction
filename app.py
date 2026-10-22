import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Enter the student's details below to predict "
    "their final grade."
)

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.joblib")

try:
    bundle = load_model()

    # Support either a saved dictionary or a direct model
    if isinstance(bundle, dict):
        model = bundle.get("model")
        feature_columns = bundle.get("feature_columns")

        if model is None or feature_columns is None:
            st.error(
                "The saved model bundle must contain "
                "'model' and 'feature_columns'."
            )
            st.stop()
    else:
        st.error(
            "The saved file must contain a dictionary "
            "with 'model' and 'feature_columns'."
        )
        st.stop()

except FileNotFoundError:
    st.error(
        "Model file not found. Keep "
        "'student_performance_model.joblib' "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

# --------------------------------------------------
# User-friendly field labels
# Original column names remain unchanged
# --------------------------------------------------
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
    "absences": "Number of School Absences"
}

# Explanations for coded values
field_help = {
    "Medu": "0 = None, 1 = Primary, 2 = Grades 5–9, "
            "3 = Secondary, 4 = Higher education",
    "Fedu": "0 = None, 1 = Primary, 2 = Grades 5–9, "
            "3 = Secondary, 4 = Higher education",
    "traveltime": "1 = Under 15 min, 2 = 15–30 min, "
                  "3 = 30–60 min, 4 = Over 60 min",
    "studytime": "1 = Under 2 hours, 2 = 2–5 hours, "
                 "3 = 5–10 hours, 4 = Over 10 hours per week",
    "failures": "Number of previous class failures: 0, 1, 2, or 3 "
                "(3 also represents three or more)",
    "famrel": "1 = Very bad, 2 = Bad, 3 = Average, "
              "4 = Good, 5 = Excellent",
    "freetime": "1 = Very low, 2 = Low, 3 = Average, "
                "4 = High, 5 = Very high",
    "goout": "1 = Very low, 2 = Low, 3 = Average, "
             "4 = High, 5 = Very high",
    "Dalc": "Weekday alcohol consumption scale: 1 = Very low "
            "and 5 = Very high",
    "Walc": "Weekend alcohol consumption scale: 1 = Very low "
            "and 5 = Very high",
    "health": "1 = Very bad, 2 = Bad, 3 = Average, "
              "4 = Good, 5 = Very good",
    "absences": "Number of school absences"
}

# Display labels for categorical values
display_labels = {
    "school": {
        "GP": "GP — Gabriel Pereira",
        "MS": "MS — Mousinho da Silveira"
    },
    "sex": {
        "F": "Female",
        "M": "Male"
    },
    "address": {
        "U": "Urban",
        "R": "Rural"
    },
    "famsize": {
        "LE3": "3 or Fewer Family Members",
        "GT3": "More Than 3 Family Members"
    },
    "Pstatus": {
        "T": "Together — Parents Live Together",
        "A": "Apart — Parents Live Separately"
    },
    "schoolsup": {"yes": "Yes", "no": "No"},
    "famsup": {"yes": "Yes", "no": "No"},
    "paid": {"yes": "Yes", "no": "No"},
    "activities": {"yes": "Yes", "no": "No"},
    "nursery": {"yes": "Yes", "no": "No"},
    "higher": {"yes": "Yes", "no": "No"},
    "internet": {"yes": "Yes", "no": "No"},
    "romantic": {"yes": "Yes", "no": "No"},
    "Mjob": {
        "teacher": "Teacher",
        "health": "Healthcare",
        "services": "Services",
        "at_home": "At Home",
        "other": "Other"
    },
    "Fjob": {
        "teacher": "Teacher",
        "health": "Healthcare",
        "services": "Services",
        "at_home": "At Home",
        "other": "Other"
    },
    "reason": {
        "home": "Close to Home",
        "reputation": "School Reputation",
        "course": "Course Preference",
        "other": "Other"
    },
    "guardian": {
        "mother": "Mother",
        "father": "Father",
        "other": "Other"
    }
}

# --------------------------------------------------
# Input form
# --------------------------------------------------
input_data = {}

# Columns are taken directly from the trained model bundle
with st.form("student_prediction_form"):
    st.subheader("Enter Student Details")

    columns = st.columns(2)

    for index, feature in enumerate(feature_columns):
        label = custom_labels.get(
            feature,
            feature.replace("_", " ").title()
        )

        help_text = field_help.get(feature)
        options_map = display_labels.get(feature)

        with columns[index % 2]:
            # Categorical features
            if options_map is not None:
                options = list(options_map.keys())

                # Use dataset values as submitted model values
                input_data[feature] = st.selectbox(
                    label,
                    options=options,
                    format_func=lambda value, m=options_map: m.get(
                        value, str(value)
                    ),
                    help=help_text,
                    key=f"input_{feature}"
                )

            # Numeric features
            else:
                if feature in ["Dalc", "Walc"]:
                    input_data[feature] = st.selectbox(
                        label,
                        options=[1, 2, 3, 4, 5],
                        format_func=lambda value: {
                            1: "1 — Very Low",
                            2: "2 — Low",
                            3: "3 — Moderate",
                            4: "4 — High",
                            5: "5 — Very High"
                        }[value],
                        help=help_text,
                        key=f"input_{feature}"
                    )

                elif feature in ["Medu", "Fedu"]:
                    input_data[feature] = st.selectbox(
                        label,
                        options=[0, 1, 2, 3, 4],
                        format_func=lambda value: {
                            0: "0 — None",
                            1: "1 — Primary Education",
                            2: "2 — Grades 5–9",
                            3: "3 — Secondary Education",
                            4: "4 — Higher Education"
                        }[value],
                        help=help_text,
                        key=f"input_{feature}"
                    )

                elif feature in [
                    "traveltime", "studytime", "famrel",
                    "freetime", "goout", "health"
                ]:
                    input_data[feature] = st.selectbox(
                        label,
                        options=[1, 2, 3, 4, 5]
                        if feature in ["famrel", "freetime",
                                       "goout", "health"]
                        else [1, 2, 3, 4],
                        help=help_text,
                        key=f"input_{feature}"
                    )

                elif feature == "failures":
                    input_data[feature] = st.selectbox(
                        label,
                        options=[0, 1, 2, 3],
                        help=help_text,
                        key=f"input_{feature}"
                    )

                elif feature == "age":
                    input_data[feature] = st.number_input(
                        label,
                        min_value=15,
                        max_value=22,
                        value=17,
                        step=1,
                        help=help_text,
                        key=f"input_{feature}"
                    )

                elif feature == "absences":
                    input_data[feature] = st.number_input(
                        label,
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        help=help_text,
                        key=f"input_{feature}"
                    )

                else:
                    input_data[feature] = st.number_input(
                        label,
                        value=0,
                        help=help_text,
                        key=f"input_{feature}"
                    )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:
    try:
        # Preserve the exact feature order and original names
        input_df = pd.DataFrame(
            [[input_data[col] for col in feature_columns]],
            columns=feature_columns
        )

        prediction = float(model.predict(input_df)[0])

        st.subheader("Prediction Result")
        st.metric(
            label="Predicted Final Grade (G3)",
            value=f"{prediction:.2f} / 20"
        )

        if prediction >= 16:
            st.success("Predicted performance: Very Good")
        elif prediction >= 14:
            st.success("Predicted performance: Good")
        elif prediction >= 10:
            st.info("Predicted performance: Satisfactory")
        else:
            st.warning(
                "Predicted performance: Needs Improvement"
            )

        st.caption(
            "This is an estimate from a machine-learning model, "
            "not a guaranteed grade or a formal student assessment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the saved model expects the same feature "
            "names and data types as the input form."
        )
