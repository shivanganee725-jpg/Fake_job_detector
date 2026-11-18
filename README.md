# Fake Job Posting Detection — Streamlit

## Project files
- `app.py` — Streamlit interface and Logistic Regression model
- `requirements.txt` — Python dependencies
- `fake_job_postings.csv` — your dataset (add this file here, or upload it in the app sidebar)

## Run on Windows
1. Install Python 3.10 or newer.
2. Open Command Prompt in this folder.
3. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Streamlit will open the app in your browser. If it does not, open the local URL printed in the terminal (usually http://localhost:8501).

## Dataset requirement
The CSV must contain these columns:
- `telecommuting`
- `has_company_logo`
- `has_questions`
- `fraudulent`

The target column must use `0` for real and `1` for fraudulent postings. You can upload the CSV from the sidebar, or put it next to `app.py` with the exact name `fake_job_postings.csv`.

## Important project limitation
The uploaded notebook uses only three numeric metadata features, not the job title or description text. Therefore this app faithfully implements that notebook, but it is a basic baseline and may miss fraud patterns. It should not be presented as a reliable real-world fraud detector.
