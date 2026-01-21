from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("calibrated_model.pkl", "rb") as f:
    model = pickle.load(f)


def create_features(form):
    tenure = int(form["tenure"])
    monthly_charges = float(form["MonthlyCharges"])
    total_charges = float(form["TotalCharges"])

    senior = int(form["SeniorCitizen"])
    dependents = int(form["Dependents"])
    paperless = int(form["PaperlessBilling"])

    contract = form["Contract"]
    internet_service = form["InternetService"]
    payment_method = form["PaymentMethod"]

    avg_monthly_charge = total_charges / (tenure + 1)
    has_internet = 0 if internet_service == "No" else 1

    is_auto_payment = 1 if payment_method in [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] else 0

    if tenure <= 6:
        tenure_bin = "0-6"
    elif tenure <= 12:
        tenure_bin = "6-12"
    elif tenure <= 24:
        tenure_bin = "12-24"
    else:
        tenure_bin = "24+"

    num_services = has_internet

    df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "avg_monthly_charge": avg_monthly_charge,
        "num_services": num_services,
        "is_auto_payment": is_auto_payment,
        "has_internet": has_internet,
        "SeniorCitizen": senior,
        "Dependents": dependents,
        "PaperlessBilling": paperless,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "tenure_bin": tenure_bin
    }])

    return df


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_df = create_features(request.form)

    # Get probability from underlying pipeline
    prob = model.estimators_[0].predict_proba(input_df)[0][1]
    prediction = int(prob >= 0.19)
    prediction = int(prob >= 0.19)

    result = "Likely to Churn" if prediction else "Not Likely to Churn"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(prob, 4)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
