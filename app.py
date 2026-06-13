from flask import Flask, render_template, request, jsonify
from flask import Response, redirect, send_file, url_for
from flask_mail import Mail, Message

import csv
import json
import io
import os
import pickle
import numpy as np
from datetime import datetime
from uuid import uuid4


app = Flask(__name__)

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)
print("MAIL USER:", app.config['MAIL_USERNAME'])
print("MAIL PASSWORD SET:", bool(app.config['MAIL_PASSWORD']))

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "loan_history.json")
CONTACT_EMAILS = [
    'mimakter.de@gmail.com',
    'hello.tanim.bd@gmail.com'
]


def load_model():
    for model_path in ("TM.pkl", "model.pkl"):
        try:
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            continue
    raise FileNotFoundError("No valid model file found: TM.pkl or model.pkl")


model = load_model()
SITE_ALGORITHM = "Random Forest Classifier"
SITE_ACCURACY = "89% prediction accuracy"


def form_value(form_data, *names, default=""):
    for name in names:
        value = form_data.get(name)
        if value not in (None, "", "--") and not str(value).strip().lower().startswith("-- select"):
            return value
    return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def map_credit_score(value):
    bands = {
        "Poor (300-579)": 450,
        "Fair (580-669)": 625,
        "Good (670-799)": 735,
        "Excellent (800+)": 825,
    }
    return bands.get(value, as_int(value, 650))


def map_existing_loans(value, other_value=""):
    if value == "Other":
        return as_int(other_value, 0)
    if value == "3+":
        return 3
    return as_int(value, 0)


def map_loan_history(value, other_value=""):
    if value == "Other":
        return as_int(other_value, 0)
    return as_int(value, 0)


def map_loan_term(value):
    if isinstance(value, str) and "Year" in value:
        return as_float(value.split()[0], 1.0)
    return as_float(value, 1.0)


def map_bank_history(value):
    score_map = {
        "Poor": 1,
        "Average": 2,
        "Good": 3,
        "Excellent": 4,
    }
    return score_map.get(value, as_int(value, 2))


def map_transaction_frequency(value):
    freq_map = {
        "Low": 10,
        "Medium": 25,
        "High": 45,
    }
    return freq_map.get(value, as_int(value, 20))


def map_default_risk(value):
    risk_map = {
        "Low Risk": 0.2,
        "Medium Risk": 0.5,
        "High Risk": 0.8,
    }
    return risk_map.get(value, as_float(value, 0.5))


def map_interest_rate(purpose, value):
    if value not in (None, ""):
        parsed = as_float(value, -1)
        if parsed >= 0:
            return parsed

    # Bangladeshi Law Interest Rates
    purpose_rate = {
        "Home": 9.5,              # Land & Building Loan
        "Personal": 13.5,         # Personal Loan
        "Vehicle": 11.0,          # Vehicle Loan
        "Business": 12.5,         # Business Loan
        "Education": 8.5,         # Education Loan
        "Agriculture": 8.0,       # Agriculture Loan
        "Other": 12.0,
    }
    return purpose_rate.get(purpose, 12.0)


def build_feature_vector(submission):
    male = 1 if submission["gender"] == "Male" else 0

    married = 1 if submission["marital_status"] == "Married" else 0
    single = 1 if submission["marital_status"] == "Single" else 0

    dep_1 = 1 if submission["dependents"] == "1" else 0
    dep_2 = 1 if submission["dependents"] == "2" else 0
    dep_3 = 1 if submission["dependents"] == "3" else 0

    high_school = 1 if submission["education"] == "High School" else 0
    postgraduate = 1 if submission["education"] == "Postgraduate" else 0

    self_employed = 1 if submission["employment_status"] == "Self-Employed" else 0
    unemployed = 1 if submission["employment_status"] == "Unemployed" else 0

    home = 1 if submission["loan_purpose"] == "Home" else 0
    personal = 1 if submission["loan_purpose"] == "Personal" else 0
    vehicle = 1 if submission["loan_purpose"] == "Vehicle" else 0

    own = 1 if submission["residential_status"] == "Own" else 0
    rent = 1 if submission["residential_status"] == "Rent" else 0

    gender_male = male

    marital_status_married = married
    marital_status_single = single

    education_high_school = high_school
    education_postgraduate = postgraduate

    employment_status_self_employed = self_employed
    employment_status_unemployed = unemployed

    loan_purpose_home = home
    loan_purpose_personal = personal
    loan_purpose_vehicle = vehicle

    residential_status_own = own
    residential_status_rent = rent

    occupation_freelancer = 1 if submission["occupation_type"] == "Freelancer" else 0
    occupation_professional = 1 if submission["occupation_type"] == "Professional" else 0
    occupation_salaried = 1 if submission["occupation_type"] == "Salaried" else 0

    city_suburban = 1 if submission["city_town"] == "Suburban" else 0
    city_urban = 1 if submission["city_town"] == "Urban" else 0

    loan_type_unsecured = 1 if submission["loan_type"] == "Unsecured" else 0

    co_applicant_yes = 1 if submission["co_applicant"] == "Yes" else 0

    return [[
        submission["age"],
        submission["monthly_expenses"],
        submission["credit_score"],
        submission["existing_loans"],
        submission["total_existing_loan_amount"],
        submission["outstanding_debt"],
        submission["loan_history"],
        submission["interest_rate"],
        submission["bank_account_history"],
        submission["transaction_frequency"],
        submission["default_risk"],
        submission["loan_to_income_ratio"],
        submission["annual_income_log"],
        submission["loan_amount_log"],
        submission["loan_term_log"],
        male,
        married,
        single,
        dep_1,
        dep_2,
        dep_3,
        high_school,
        postgraduate,
        self_employed,
        unemployed,
        home,
        personal,
        vehicle,
        own,
        rent,
        gender_male,
        marital_status_married,
        marital_status_single,
        dep_1,
        dep_2,
        dep_3,
        education_high_school,
        education_postgraduate,
        employment_status_self_employed,
        employment_status_unemployed,
        loan_purpose_home,
        loan_purpose_personal,
        loan_purpose_vehicle,
        residential_status_own,
        residential_status_rent,
        occupation_freelancer,
        occupation_professional,
        occupation_salaried,
        city_suburban,
        city_urban,
        loan_type_unsecured,
        co_applicant_yes,
    ]]


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as history_file:
            history = json.load(history_file)
            if isinstance(history, list):
                return history
    except (OSError, json.JSONDecodeError):
        pass

    return []


def save_history(history_entries):
    with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
        json.dump(history_entries, history_file, indent=2, ensure_ascii=False)


def remove_history_entry(application_id):
    history_entries = load_history()
    updated_history = [entry for entry in history_entries if entry.get("application_id") != application_id]
    save_history(updated_history)
    return updated_history


def history_export_rows(history_entries):
    rows = []
    for entry in history_entries:
        form_data = entry.get("form", {})
        row = {
            "application_id": entry.get("application_id", ""),
            "submitted_at": entry.get("submitted_at", ""),
            "decision": entry.get("decision", ""),
            "score": entry.get("score", ""),
            "summary": entry.get("summary", ""),
            "reasons": " | ".join(entry.get("reasons", [])),
        }
        row.update(form_data)
        rows.append(row)
    return rows


def evaluate_application(data):
    """Deprecated manual scoring path.

    Kept only for reference. Prediction is now performed with the trained model.
    """
    pass


def normalize_history_record(data, decision, summary, reasons, score, risk_level="Medium Risk"):
    return {
        "application_id": uuid4().hex[:8].upper(),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision": decision,
        "summary": summary,
        "score": score,
        "risk_level": risk_level,
        "reasons": reasons,
        "form": data,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    history_entries = load_history()

    # Calculate statistics
    approved_count = sum(1 for entry in history_entries if entry.get("decision") == "Approved")
    rejected_count = sum(1 for entry in history_entries if entry.get("decision") == "Rejected")
    total_count = len(history_entries)

    # Get last_result from query param if redirected from POST
    last_result = None
    result_id = request.args.get('result_id')
    if result_id:
        for entry in history_entries:
            if entry.get('application_id') == result_id:
                last_result = entry
                break

    if request.method == "POST":
        loan_purpose_selected = form_value(request.form, "loan_purpose")
        loan_purpose_other = form_value(request.form, "loan_purpose_other")
        if loan_purpose_selected == "Other" and loan_purpose_other:
            loan_purpose_text = loan_purpose_other
            normalized_loan_purpose = "Other"
        else:
            loan_purpose_text = loan_purpose_selected
            normalized_loan_purpose = loan_purpose_selected

        existing_loans_selected = form_value(request.form, "existing_loans")
        existing_loans_other = form_value(request.form, "existing_loans_other")

        loan_history_selected = form_value(request.form, "loan_history")
        loan_history_other = form_value(request.form, "loan_history_other")

        submission = {
            "age": as_int(form_value(request.form, "age")),
            "gender": form_value(request.form, "gender"),
            "marital_status": form_value(request.form, "marital_status"),
            "dependents": form_value(request.form, "dependents"),
            "education": form_value(request.form, "education"),
            "employment_status": form_value(request.form, "employment_status"),
            "occupation_type": form_value(request.form, "occupation_type"),
            "monthly_expenses": as_float(form_value(request.form, "monthly_expenses")),
            "annual_income": as_float(form_value(request.form, "annual_income", "ApplicantIncome")),
            "credit_score": map_credit_score(form_value(request.form, "credit_score", "credit")),
            "existing_loans": map_existing_loans(existing_loans_selected, existing_loans_other),
            "existing_loans_label": existing_loans_selected,
            "existing_loans_other": existing_loans_other,
            "outstanding_debt": as_float(form_value(request.form, "outstanding_debt")),
            "total_existing_loan_amount": as_float(form_value(request.form, "total_existing_loan_amount")),
            "loan_amount": as_float(form_value(request.form, "loan_amount", "LoanAmount")),
            "loan_term": map_loan_term(form_value(request.form, "loan_term", "Loan_Amount_Term")),
            "loan_purpose": normalized_loan_purpose,
            "loan_purpose_text": loan_purpose_text,
            "loan_type": form_value(request.form, "loan_type"),
            "interest_rate": map_interest_rate(normalized_loan_purpose, form_value(request.form, "interest_rate")),
            "loan_history": map_loan_history(loan_history_selected, loan_history_other),
            "loan_history_label": loan_history_selected,
            "loan_history_other": loan_history_other,
            "bank_account_history": map_bank_history(form_value(request.form, "bank_account_history")),
            "transaction_frequency": map_transaction_frequency(form_value(request.form, "transaction_frequency")),
            "default_risk": map_default_risk(form_value(request.form, "default_risk")),
            "co_applicant": form_value(request.form, "co_applicant"),
            "residential_status": form_value(request.form, "residential_status"),
            "city_town": form_value(request.form, "city_town"),
        }

        annual_income = submission["annual_income"]
        loan_amount = submission["loan_amount"]
        loan_term = submission["loan_term"]

        submission["annual_income_log"] = float(np.log1p(annual_income)) if annual_income > 0 else 0.0
        submission["loan_amount_log"] = float(np.log1p(loan_amount)) if loan_amount > 0 else 0.0
        submission["loan_term_log"] = float(np.log1p(loan_term)) if loan_term > 0 else 0.0
        submission["loan_to_income_ratio"] = round((loan_amount / annual_income), 4) if annual_income > 0 else 0.0

        features = build_feature_vector(submission)
        prediction = model.predict(features)[0]

        if prediction == 1:
            decision = "Approved"
        else:
            decision = "Rejected"

        score = round(max(model.predict_proba(features)[0]) * 100, 2)
        
        # Generate risk level
        if score >= 80:
            risk_level = "Low Risk"
        elif score >= 60:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        summary = f"Approval Probability: {score}%"
        
        # Generate detailed reasons
        reasons = []
        
        # Check credit score
        if submission["credit_score"] < 500:
            reasons.append("Low credit score may impact approval")
        elif submission["credit_score"] > 750:
            reasons.append("Excellent credit score supports approval")
        
        # Check outstanding debt
        if submission["outstanding_debt"] > 50000:
            reasons.append("High outstanding debt may impact approval")
        elif submission["outstanding_debt"] == 0:
            reasons.append("No existing debt is favorable")
        
        # Check loan to income ratio
        if submission["loan_to_income_ratio"] > 5:
            reasons.append("High loan-to-income ratio")
        elif submission["loan_to_income_ratio"] < 1:
            reasons.append("Favorable loan-to-income ratio")
        
        # Check existing loans
        if submission["existing_loans"] >= 3:
            reasons.append("Multiple active loans may affect approval")
        
        # Check annual income
        if submission["annual_income"] < 20000:
            reasons.append("Lower annual income noted")
        elif submission["annual_income"] > 100000:
            reasons.append("Strong annual income supports approval")
        
        # Check employment status
        if submission["employment_status"] == "Unemployed":
            reasons.append("Unemployed status may impact approval")
        elif submission["employment_status"] == "Self-Employed":
            reasons.append("Self-employed income stability reviewed")
        
        # Check default risk
        if submission["default_risk"] > 0.7:
            reasons.append("High default risk detected")
        elif submission["default_risk"] < 0.3:
            reasons.append("Low default risk is favorable")
        
        # Add default reasons if none generated
        if not reasons:
            if decision == "Approved":
                reasons.append("Application meets all lending criteria")
                reasons.append("Financial profile is acceptable")
            else:
                reasons.append("Application does not meet lending criteria")
                reasons.append("Risk factors exceed acceptable thresholds")
        
        prediction_text = f"{decision}\n{summary}"

        history_record = normalize_history_record(submission, decision, summary, reasons, score, risk_level)
        history_entries.insert(0, history_record)
        save_history(history_entries)

        # Redirect to avoid form resubmission on page reload
        return redirect(url_for('predict', result_id=history_record['application_id']))

    return render_template(
        "prediction.html", 
        history_entries=history_entries, 
        approved_count=approved_count, 
        rejected_count=rejected_count, 
        total_count=total_count,
        last_result=last_result
    )


@app.route("/history/delete/<application_id>", methods=["POST"])
def delete_history_item(application_id):
    history_entries = remove_history_entry(application_id)
    approved_count = sum(1 for entry in history_entries if entry.get("decision") == "Approved")
    rejected_count = sum(1 for entry in history_entries if entry.get("decision") == "Rejected")
    total_count = len(history_entries)
    return jsonify({
        "status": "success",
        "message": "History item deleted successfully.",
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "total_count": total_count
    })


@app.route("/history/export/<file_format>")
def export_history(file_format):
    history_entries = load_history()
    safe_format = file_format.lower()

    if safe_format == "csv":
        output = io.StringIO()
        rows = history_export_rows(history_entries)
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        csv_bytes = io.BytesIO(output.getvalue().encode("utf-8-sig"))
        csv_bytes.seek(0)
        return send_file(
            csv_bytes,
            mimetype="text/csv",
            as_attachment=True,
            download_name="loan_history.csv",
        )

    json_bytes = io.BytesIO(json.dumps(history_entries, indent=2, ensure_ascii=False).encode("utf-8"))
    json_bytes.seek(0)
    return send_file(
        json_bytes,
        mimetype="application/json",
        as_attachment=True,
        download_name="loan_history.json",
    )


@app.route('/download_reports')
def download_reports():
    """Render the PDF template for one or multiple history entries and auto-download.
    Optional query param: application_id to render a single entry.
    """
    history_entries = load_history()
    application_id = request.args.get('application_id')

    if application_id:
        entries = [e for e in history_entries if e.get('application_id') == application_id]
    else:
        entries = history_entries

    approved_count = sum(1 for entry in history_entries if entry.get("decision") == "Approved")
    rejected_count = sum(1 for entry in history_entries if entry.get("decision") == "Rejected")
    total_count = len(history_entries)

    members = [
        {"name": "Saifulla Tanim", "role": "Developer"},
        {"name": "Mim Akter", "role": "Developer"},
    ]

    return render_template(
        'pdf_template.html',
        entries=entries or [],
        members=members,
        approved_count=approved_count,
        rejected_count=rejected_count,
        total_count=total_count,
        site_algorithm=SITE_ALGORITHM,
        site_accuracy=SITE_ACCURACY,
        generated_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        year=datetime.utcnow().year,
        auto_download=True,
    )


@app.route("/contact", methods=["POST"])
def send_contact_email():
    """Handle contact form submissions and send emails"""
    try:
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip()
        message_text = request.form.get("message", "").strip()
        
        if not all([first_name, last_name, email, message_text]):
            return {"status": "error", "message": "Please fill all required fields"}, 400
        
        # Create email message
        subject = f"New Contact Message from {first_name} {last_name}"
        
        email_body = f"""
New Contact Message Received:

Name: {first_name} {last_name}
Email: {email}
Date & Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Message:
{message_text}

---
This is an automated message from the Loan Prediction System.
Please reply directly to the sender's email address: {email}
"""
        
        # Send to both contact emails
        for recipient in CONTACT_EMAILS:
            try:
                msg = Message(
                    subject=subject,
                    recipients=[recipient],
                    body=email_body,
                    reply_to=email
                )
                mail.send(msg)
            except Exception as e:
                print(f"Error sending email to {recipient}: {str(e)}")
        
        # Send confirmation email to sender
        try:
            confirmation_msg = Message(
                subject="Message Received - Loan Prediction System",
                recipients=[email],
                body=f"""
Hello {first_name},

Thank you for contacting us. We have successfully received your message and will respond as soon as possible.

Best regards,
Loan Prediction System Team

Contact Emails:
- mimakter.de@gmail.com
- hello.tanim.bd@gmail.com
"""
            )
            mail.send(confirmation_msg)
        except Exception as e:
            print(f"Error sending confirmation email: {str(e)}")
        
        return {"status": "success", "message": "Your message has been sent successfully!"}, 200
    
    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}, 500


@app.route('/debug/send-test')
def debug_send_test():
    """Debug route: attempt to send a test email to the configured contact addresses
    and return any exception text so we can diagnose deployment SMTP issues.
    Remove or protect this route in production.
    """
    subject = "[Debug] Test Email from Loan Prediction System"
    body = "This is a test message from the deployed Loan Prediction System (debug route)."
    try:
        msg = Message(subject=subject, recipients=CONTACT_EMAILS, body=body)
        mail.send(msg)
        return {"status": "success", "message": f"Test email sent to: {CONTACT_EMAILS}"}, 200
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Debug send-test failed:\n", tb)
        return {"status": "error", "message": str(e), "trace": tb}, 500


if __name__ == "__main__":
    app.run(debug=True)