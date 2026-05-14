from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

print('Model loaded successfully')

# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

# ==========================================
# PREDICT API
# ==========================================


@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ==========================================
        # READ JSON PAYLOAD
        # ==========================================

        data = request.get_json()

        print(data)

        # ==========================================
        # CREATE DATAFRAME
        # ==========================================

        df = pd.DataFrame([data])

        # ==========================================
        # NUMERIC DATA
        # ==========================================

        df_numeric = df.select_dtypes(include=[np.number]).fillna(0)

        # ==========================================
        # SCALE
        # ==========================================

        X_scaled = scaler.transform(df_numeric)

        # ==========================================
        # PCA
        # ==========================================

        X_pca = pca.transform(X_scaled)

        # ==========================================
        # PREDICT
        # ==========================================

        prediction = model.predict(X_pca)[0]

        # Isolation Forest anomaly score
        anomaly_score = model.decision_function(X_pca)[0]

        # Convert score to fraud percentage
        fraud_score = round((1 - anomaly_score) * 100, 2)

        # Limit between 0 and 100
        fraud_score = max(0, min(fraud_score, 100))

        # ==========================================
        # FRAUD LABEL
        # ==========================================

        fraud_label = (
            'Fraud'
            if prediction == -1
            else 'Normal'
        )

        # ==========================================
        # ANOMALY SCORE
        # ==========================================

        anomaly_score = model.decision_function(X_pca)[0]

        raw_score = round((1 - anomaly_score) * 100, 2)

        # ==========================================
        # FRAUD SCORE NORMALIZATION
        # ==========================================

        if fraud_label == 'Fraud':

            fraud_score = max(65, min(raw_score, 99))

        else:

            fraud_score = max(5, min(raw_score, 35))

        fraud_score = round(fraud_score, 2)

        # ==========================================
        # RISK LEVEL
        # ==========================================

        if fraud_label == 'Fraud':

            if fraud_score >= 85:
                risk_level = 'High Risk'

            else:
                risk_level = 'Medium Risk'

        else:

            risk_level = 'Low Risk'

        # ==========================================
        # ANOMALY SEVERITY
        # ==========================================

        if fraud_score >= 85:
            anomaly_severity = 'Critical'

        elif fraud_score >= 65:
            anomaly_severity = 'High'

        elif fraud_score >= 40:
            anomaly_severity = 'Moderate'

        else:
            anomaly_severity = 'Low'

        # ==========================================
        # ESTIMATED FINANCIAL RISK
        # ==========================================

        estimated_financial_risk = round(
            data['IP_DRG_QUINT_PMT_AVG']
            * (fraud_score / 100),
            2
        )

        # ==========================================
        # CLAIM COMPLEXITY
        # ==========================================

        complexity_score = 0

        if data['IP_CLM_DAYS_CD'] >= 3:
            complexity_score += 1

        if data['IP_CLM_BASE_DRG_CD'] in [470, 871, 291]:
            complexity_score += 1

        if data['IP_CLM_ICD9_PRCDR_CD'] in [36, 81]:
            complexity_score += 1

        if complexity_score >= 3:
            claim_complexity = 'Highly Complex'

        elif complexity_score >= 2:
            claim_complexity = 'Complex'

        else:
            claim_complexity = 'Standard'

        # ==========================================
        # CONFIDENCE LEVEL
        # ==========================================

        confidence_level = round(
            100 - abs(anomaly_score * 100),
            2
        )

        confidence_level = max(
            60,
            min(confidence_level, 99)
        )

        # ==========================================
        # FRAUD INDICATORS
        # ==========================================

        fraud_indicators = []

        if data['IP_CLM_DAYS_CD'] >= 4:
            fraud_indicators.append(
                'Extended hospital stay'
            )

        if data['IP_DRG_QUINT_PMT_CD'] >= 4:
            fraud_indicators.append(
                'High payment category'
            )

        if data['IP_CLM_BASE_DRG_CD'] == 871:
            fraud_indicators.append(
                'High-risk DRG category'
            )

        if data['IP_CLM_ICD9_PRCDR_CD'] in [36, 81]:
            fraud_indicators.append(
                'Complex surgical procedure'
            )

        if fraud_score >= 80:
            fraud_indicators.append(
                'Strong anomaly pattern detected'
            )

        if len(fraud_indicators) == 0:

            fraud_indicators.append(
                'No major anomaly indicators'
            )

        # ==========================================
        # RECOMMENDED ACTION
        # ==========================================

        if fraud_label == 'Fraud':

            recommended_action = (
                'Manual Investigation Required'
            )

        elif fraud_score > 40:

            recommended_action = (
                'Secondary Verification Recommended'
            )

        else:

            recommended_action = (
                'Approve Claim'
            )

        # ==========================================
        # FINAL RESPONSE
        # ==========================================

        response = data

        response['fraud_label'] = fraud_label
        response['fraud_score'] = fraud_score
        response['risk_level'] = risk_level
        response['anomaly_severity'] = anomaly_severity
        response['estimated_financial_risk'] = estimated_financial_risk
        response['claim_complexity'] = claim_complexity
        response['confidence_level'] = confidence_level
        response['recommended_action'] = recommended_action
        response['fraud_indicators'] = fraud_indicators

        return jsonify(response)

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500
# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )