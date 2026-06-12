**Loan Prediction (Flask + ML)**

- **Project**: Loan approval prediction web app using a pre-trained ML model with a Flask frontend and JSON-based history persistence.

**Tech Stack**

- **Language**: Python 3
- **Web**: Flask, Jinja2 templates
- **ML**: scikit-learn (pre-trained model pickle)
- **Frontend**: Tailwind-like utility classes + Bootstrap, Chart.js, html2pdf.js
- **Email**: Flask-Mail (SMTP)
- **Persistence**: JSON file (`loan_history.json`)
- **Other**: numpy, html2pdf, uuid, datetime

**High-level Features**

- Predict loan approval using a pre-trained model (`TM.pkl` or `model.pkl`).
- Result summary UI showing Decision, Confidence, Risk Level and Key Decision Factors.
- Download PDF report for a single result (server-side `pdf_template.html`) and client-side export for result card.
- History of applications saved in `loan_history.json` with download and delete (AJAX) support.
- Export history as CSV/JSON.
- Contact form that sends email notifications (configured via `EMAIL_SETUP.md`).

**Repository Structure** (key files)

- **app.py**: Main Flask application and routes
- **requirements.txt**: Python dependencies
- **train.csv**: Training data (if present)
- **TM.pkl / model.pkl**: Pre-trained ML model (required at runtime)
- **loan_history.json**: Stored application history (created after first prediction)
- **templates/**: Jinja2 templates
  - **[templates/prediction.html](templates/prediction.html)**: Main app UI and scripts
  - **[templates/pdf_template.html](templates/pdf_template.html)**: Server-side PDF report template
  - **[templates/index.html](templates/index.html)**: Landing page
  - **[templates/prediction.html](templates/prediction.html)**: Prediction page (form + history)
- **EMAIL_SETUP.md**: Instructions for configuring SMTP/Flask-Mail

**Important Routes / Endpoints**

- **GET /** — Landing page ([templates/index.html](templates/index.html)).
- **GET /predict** — Show prediction form and history, optionally displays `?result_id=` after a redirect.
- **POST /predict** — Submit form to run prediction; uses redirect-after-post to avoid duplicate history on reload.
- **POST /history/delete/<application_id>** — Delete a history entry (returns JSON for AJAX updates).
- **GET /download_reports** — Render `pdf_template.html` for one or many entries. Optional query param: `application_id` to generate a single-report download.
- **GET /history/export/<file_format>** — Export history as JSON or CSV.
- **POST /contact** — Send contact email via Flask-Mail.

**Model & Data**

- Place the pre-trained model file (`TM.pkl` or `model.pkl`) in the project root so `app.py` can load it.
- The app persists application history to `loan_history.json`. If the file is missing, the app creates it on first prediction.

**Local Setup & Run (Windows)**

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure the model file (`TM.pkl` or `model.pkl`) is present in the project root.
4. (Optional) Configure SMTP for contact emails — see [EMAIL_SETUP.md](EMAIL_SETUP.md).
5. Run the app:

```powershell
python app.py
```

6. Open the app in your browser: http://127.0.0.1:5000/predict

**Notes & Tips**

- The predict route uses redirect-after-post to prevent duplicate history entries when the user reloads the result page.
- Deleting history items is done via AJAX and the backend returns JSON with updated `approved_count`, `rejected_count`, and `total_count` so the UI updates without a full reload.
- There are two PDF mechanisms:
  - Server-side: `/download_reports?application_id=<id>` renders `templates/pdf_template.html` and triggers auto-download.
  - Client-side: `html2pdf.js` is used for quick report export from the result card (this is implemented in the page script as `exportPDF()`).
- If you want to run the app on Linux/macOS, activate virtualenv with `source .venv/bin/activate` instead of PowerShell activation commands.

**Testing Quick Checklist**

- Submit a full form at `/predict` and confirm a result card appears.
- Reload the page — history should not duplicate the last entry.
- Click `Download PDF Report` on the result card — server-side PDF should open/download for that application.
- Delete an entry from history — it should be removed without a full page refresh and stats should update.
- Try exporting history as CSV/JSON via the top-right buttons on the History card.

**Contributing / Next Improvements**

- Add model explainability (SHAP) to show feature contributions per prediction.
- Add authentication for an admin dashboard to review and filter history.
- Add server-side PDF generation fallback (wkhtmltopdf) for higher fidelity reports.

---

If you want, I can also:

- Add a short `RUN.md` with screenshots and common troubleshooting steps.
- Create a `Procfile` / Dockerfile for deployment.
