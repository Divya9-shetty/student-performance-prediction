# Student Performance Prediction — Phase 3

## Project
Student Performance Prediction Using Machine Learning

## Files
- `Phase2_Model_Comparison.ipynb`: Phase 2 model implementation and evaluation.
- `student-performance-combined.csv`: combined Math and Portuguese course records used in the notebook.
- `train_model.py`: fits the selected Random Forest pipeline and saves the deployment artifact.
- `app.py`: Streamlit user interface.
- `requirements.txt`: Python dependencies.
- `Phase3_Project_Report.docx`: report draft based on the Phase 2 notebook's reported metrics.

## Run locally (Windows)
1. Install Python 3.10+.
2. Open Command Prompt in this folder.
3. Create a virtual environment (recommended):
   `python -m venv .venv`
4. Activate it:
   `.venv\Scripts\activate`
5. Install packages:
   `python -m pip install -r requirements.txt`
6. Train and save the model:
   `python train_model.py`
7. Start the web app:
   `python -m streamlit run app.py`
8. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Demonstration checklist
- Start the app.
- Show the model name.
- Enter a student profile.
- Click **Predict Final Grade**.
- Explain that G3 is the predicted final grade out of 20.
- Capture a screenshot of the running app and add it to the final report.

## Methodology notes
The Phase 2 notebook chose Random Forest based on the lowest five-fold CV RMSE (3.179). Gradient Boosting had a slightly lower test RMSE (3.623 vs 3.673), so this selection criterion must be described clearly. The deployment script refits the chosen Random Forest pipeline on all available records after evaluation.

The current CSV has 33 columns and no `subject` column. This matches the uploaded Phase 2 notebook. The 1,044 rows are course records, not necessarily unique students. Students can occur in both subject datasets; random row splitting can therefore cause overlap between training and test data. Report this limitation.
