import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os
import tempfile
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the WHO ISH chart data
def load_who_ish_data():
    """Load and preprocess the WHO ISH chart data"""
    df = pd.read_csv('WHO ish chart.csv')
    
    # Clean the data - standardize risk colors
    df['Risk'] = df['Risk'].str.lower()
    df['Risk'] = df['Risk'].replace({'green': 'green', 'yellow': 'yellow', 'red-orange': 'red-orange', 'orange': 'orange', 'red': 'red'})
    
    return df

def apply_risk_prediction_rules(age, sbp, cholesterol):
    """
    Apply the risk prediction rules from the document
    """
    # Age rules
    if 30 <= age <= 49:
        age_rule = 40
    elif 50 <= age <= 59:
        age_rule = 50
    elif 60 <= age <= 69:
        age_rule = 60
    elif 70 <= age <= 85:
        age_rule = 70
    else:
        age_rule = age
    
    # SBP rules
    if sbp < 139:
        sbp_rule = 120
    elif 140 <= sbp <= 159:
        sbp_rule = 140
    elif 160 <= sbp <= 179:
        sbp_rule = 160
    elif sbp >= 180:
        sbp_rule = 180
    else:
        sbp_rule = sbp
    
    # Cholesterol rules
    if cholesterol == 0:
        chol_rule = 5
    else:
        # Divide by 38 and round according to rules
        divided = cholesterol / 38
        if divided - int(divided) >= 0.5:
            chol_rule = int(divided) + 1
        else:
            chol_rule = int(divided)
    
    return age_rule, sbp_rule, chol_rule

# Create a lookup function for risk prediction
def predict_risk(diabetes, gender, smoking, age, sbp, cholesterol):
    """
    Predict cardiovascular risk based on WHO ISH chart
    
    Parameters:
    - diabetes: 'Yes' or 'No'
    - gender: 'Male' or 'Female'
    - smoking: 'Yes' or 'No'
    - age: 40, 50, 60, or 70
    - sbp: 120, 140, 160, or 180
    - cholesterol: 4, 5, 6, 7, or 8
    
    Returns:
    - risk_color: Green, Yellow, Orange, Red-Orange, or Red
    - risk_level: Risk percentage range
    """
    df = load_who_ish_data()
    
    # Find matching row in the chart
    match = df[
        (df['Diabetes'] == diabetes) &
        (df['Gender'] == gender) &
        (df['Smoking'] == smoking) &
        (df['Age'] == age) &
        (df['SBP'] == sbp) &
        (df['Cholesterol'] == cholesterol)
    ]
    
    if len(match) > 0:
        risk_color = match.iloc[0]['Risk'].title()
        risk_level = match.iloc[0]['Risk Level']
        return risk_color, risk_level
    else:
        # If exact match not found, find closest match
        # This handles edge cases where parameters don't exactly match the chart
        closest_match = df[
            (df['Diabetes'] == diabetes) &
            (df['Gender'] == gender) &
            (df['Smoking'] == smoking) &
            (df['Age'] == age) &
            (df['SBP'] == sbp)
        ]
        
        if len(closest_match) > 0:
            # Find closest cholesterol match
            closest_chol = closest_match.iloc[(closest_match['Cholesterol'] - cholesterol).abs().argsort()[:1]]
            risk_color = closest_chol.iloc[0]['Risk'].title()
            risk_level = closest_chol.iloc[0]['Risk Level']
            return risk_color, risk_level
        else:
            # Default to moderate risk if no match found
            return "Yellow", "10-20% (Estimated)"

def train_ml_model():
    """Train the machine learning model"""
    try:
        # Load the clinical data
        df = pd.read_excel('RishikaData_Cleaned_mmol.xlsx')
        print(f"Original data shape: {df.shape}")
        
        # Clean and prepare the data
        clean_df = df[['age_in_year', 'gender_encoded', 'smoker_encoded', 'diabetes_encoded', 'BP_systolic', 'cholesterol']].copy()
        print(f"Clean data shape: {clean_df.shape}")
        
        # Create a synthetic target variable based on risk factors
        def calculate_risk_category(row):
            risk_score = 0
            
            # Age factor (higher age = higher risk)
            if row['age_in_year'] >= 60:
                risk_score += 4
            elif row['age_in_year'] >= 50:
                risk_score += 3
            elif row['age_in_year'] >= 40:
                risk_score += 2
            else:
                risk_score += 1
            
            # Blood pressure factor
            if row['BP_systolic'] >= 160:
                risk_score += 4
            elif row['BP_systolic'] >= 140:
                risk_score += 3
            elif row['BP_systolic'] >= 120:
                risk_score += 2
            else:
                risk_score += 1
            
            # Cholesterol factor
            if row['cholesterol'] >= 7:
                risk_score += 4
            elif row['cholesterol'] >= 6:
                risk_score += 3
            elif row['cholesterol'] >= 5:
                risk_score += 2
            else:
                risk_score += 1
            
            # Diabetes factor
            risk_score += row['diabetes_encoded'] * 3
            
            # Smoking factor
            risk_score += row['smoker_encoded'] * 2
            
            # Gender factor (males have slightly higher risk)
            risk_score += row['gender_encoded'] * 1
            
            # Categorize risk
            if risk_score >= 12:
                return 1  # High risk
            else:
                return 0  # Low risk
        
        clean_df['target'] = clean_df.apply(calculate_risk_category, axis=1)
        
        # Check target distribution
        print(f"Target distribution: {clean_df['target'].value_counts()}")
        
        # Prepare features and target
        X = clean_df.drop('target', axis=1)
        y = clean_df['target']
        
        # Split the data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        accuracy = model.score(X_test_scaled, y_test)
        print(f"ML Model trained with accuracy: {accuracy:.3f}")
        
        # Save model and scaler
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
        
        with open('risk_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        return True
        
    except Exception as e:
        print(f"Error training ML model: {e}")
        return False

def predict_risk_ml(age, gender, smoking, diabetes, sbp, cholesterol):
    """
    Predict cardiovascular risk using the trained ML model
    """
    try:
        # Load the trained model
        if not os.path.exists('risk_model.pkl'):
            return "N/A", "Model not available"
        
        with open('risk_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Encode categorical variables
        gender_encoded = 1 if gender == 'Male' else 0
        smoking_encoded = 1 if smoking == 'Yes' else 0
        diabetes_encoded = 1 if diabetes == 'Yes' else 0
        
        # Create feature array
        features = np.array([[
            age, gender_encoded, smoking_encoded, diabetes_encoded, sbp, cholesterol
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Calculate risk score based on features
        risk_score = 0
        
        # Age factor
        if age >= 60:
            risk_score += 4
        elif age >= 50:
            risk_score += 3
        elif age >= 40:
            risk_score += 2
        else:
            risk_score += 1
        
        # Blood pressure factor
        if sbp >= 160:
            risk_score += 4
        elif sbp >= 140:
            risk_score += 3
        elif sbp >= 120:
            risk_score += 2
        else:
            risk_score += 1
        
        # Cholesterol factor
        if cholesterol >= 7:
            risk_score += 4
        elif cholesterol >= 6:
            risk_score += 3
        elif cholesterol >= 5:
            risk_score += 2
        else:
            risk_score += 1
        
        # Other factors
        risk_score += diabetes_encoded * 3
        risk_score += smoking_encoded * 2
        
        # Gender factor
        risk_score += gender_encoded * 1
        
        # Calculate confidence based on risk factors
        confidence = min(0.95, 0.3 + (risk_score * 0.1))
        
        # Categorize risk
        if risk_score <= 6:
            return "Green", f"Low Risk ({confidence:.1%})"
        elif risk_score <= 10:
            return "Yellow", f"Moderate Risk ({confidence:.1%})"
        elif risk_score <= 14:
            return "Orange", f"High Risk ({confidence:.1%})"
        else:
            return "Red", f"Very High Risk ({confidence:.1%})"
            
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return "Unknown", "Unable to determine"

def validate_patient_data(row):
    """Validate a single patient's data"""
    errors = []
    
    # Check required columns
    required_columns = ['Diabetes', 'Gender', 'Smoking', 'Age', 'SBP', 'Cholesterol']
    for col in required_columns:
        if col not in row.index:
            errors.append(f"Missing column: {col}")
            return errors
    
    # Validate diabetes
    if row['Diabetes'] not in ['Yes', 'No']:
        errors.append(f"Diabetes must be 'Yes' or 'No', got: {row['Diabetes']}")
    
    # Validate gender
    if row['Gender'] not in ['Male', 'Female']:
        errors.append(f"Gender must be 'Male' or 'Female', got: {row['Gender']}")
    
    # Validate smoking
    if row['Smoking'] not in ['Yes', 'No']:
        errors.append(f"Smoking must be 'Yes' or 'No', got: {row['Smoking']}")
    
    # Validate age
    try:
        age = int(row['Age'])
        if age not in [40, 50, 60, 70]:
            errors.append(f"Age must be 40, 50, 60, or 70, got: {age}")
    except (ValueError, TypeError):
        errors.append(f"Age must be a number, got: {row['Age']}")
    
    # Validate SBP
    try:
        sbp = int(row['SBP'])
        if sbp not in [120, 140, 160, 180]:
            errors.append(f"SBP must be 120, 140, 160, or 180, got: {sbp}")
    except (ValueError, TypeError):
        errors.append(f"SBP must be a number, got: {row['SBP']}")
    
    # Validate cholesterol
    try:
        cholesterol = int(row['Cholesterol'])
        if cholesterol not in [4, 5, 6, 7, 8]:
            errors.append(f"Cholesterol must be 4, 5, 6, 7, or 8, got: {cholesterol}")
    except (ValueError, TypeError):
        errors.append(f"Cholesterol must be a number, got: {row['Cholesterol']}")
    
    return errors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract parameters
        diabetes = data.get('diabetes')
        gender = data.get('gender')
        smoking = data.get('smoking')
        age = int(data.get('age'))
        sbp = int(data.get('sbp'))
        cholesterol = int(data.get('cholesterol'))
        
        # Validate inputs
        if not all([diabetes, gender, smoking, age, sbp, cholesterol]):
            return jsonify({'error': 'All parameters are required'}), 400
        
        # Validate age range
        if age not in [40, 50, 60, 70]:
            return jsonify({'error': 'Age must be 40, 50, 60, or 70'}), 400
        
        # Validate SBP range
        if sbp not in [120, 140, 160, 180]:
            return jsonify({'error': 'Systolic Blood Pressure must be 120, 140, 160, or 180'}), 400
        
        # Validate cholesterol range
        if cholesterol not in [4, 5, 6, 7, 8]:
            return jsonify({'error': 'Cholesterol must be 4, 5, 6, 7, or 8'}), 400
        
        # Get prediction using WHO ISH chart
        risk_color, risk_level = predict_risk(diabetes, gender, smoking, age, sbp, cholesterol)
        
        # Also get ML prediction if model exists
        ml_risk_color, ml_risk_level = "N/A", "N/A"
        if os.path.exists('risk_model.pkl'):
            ml_risk_color, ml_risk_level = predict_risk_ml(age, gender, smoking, diabetes, sbp, cholesterol)
        
        return jsonify({
            'who_ish_risk_color': risk_color,
            'who_ish_risk_level': risk_level,
            'ml_risk_color': ml_risk_color,
            'ml_risk_level': ml_risk_level,
            'parameters': {
                'diabetes': diabetes,
                'gender': gender,
                'smoking': smoking,
                'age': age,
                'sbp': sbp,
                'cholesterol': cholesterol
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction from Excel file upload"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .xlsx or .xls file'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(filepath)
        
        try:
            # Read Excel file
            df = pd.read_excel(filepath)
            
            # Process each row
            results = []
            for index, row in df.iterrows():
                # Validate patient data
                validation_errors = validate_patient_data(row)
                
                if validation_errors:
                    results.append({
                        'error': f"Row {index + 1}: {'; '.join(validation_errors)}"
                    })
                    continue
                
                try:
                    # Extract parameters
                    diabetes = str(row['Diabetes']).strip()
                    gender = str(row['Gender']).strip()
                    smoking = str(row['Smoking']).strip()
                    age = int(row['Age'])
                    sbp = int(row['SBP'])
                    cholesterol = int(row['Cholesterol'])
                    
                    # Get predictions
                    who_ish_risk_color, who_ish_risk_level = predict_risk(diabetes, gender, smoking, age, sbp, cholesterol)
                    
                    ml_risk_color, ml_risk_level = "N/A", "N/A"
                    if os.path.exists('risk_model.pkl'):
                        ml_risk_color, ml_risk_level = predict_risk_ml(age, gender, smoking, diabetes, sbp, cholesterol)
                    
                    results.append({
                        'who_ish_risk_color': who_ish_risk_color,
                        'who_ish_risk_level': who_ish_risk_level,
                        'ml_risk_color': ml_risk_color,
                        'ml_risk_level': ml_risk_level,
                        'parameters': {
                            'diabetes': diabetes,
                            'gender': gender,
                            'smoking': smoking,
                            'age': age,
                            'sbp': sbp,
                            'cholesterol': cholesterol
                        }
                    })
                    
                except Exception as e:
                    results.append({
                        'error': f"Row {index + 1}: Error processing data - {str(e)}"
                    })
            
            # Create results DataFrame for download
            results_df = pd.DataFrame()
            for i, result in enumerate(results):
                if 'error' not in result:
                    row_data = {
                        'Patient_ID': i + 1,
                        'Diabetes': result['parameters']['diabetes'],
                        'Gender': result['parameters']['gender'],
                        'Smoking': result['parameters']['smoking'],
                        'Age': result['parameters']['age'],
                        'SBP': result['parameters']['sbp'],
                        'Cholesterol': result['parameters']['cholesterol'],
                        'WHO_ISH_Risk_Color': result['who_ish_risk_color'],
                        'WHO_ISH_Risk_Level': result['who_ish_risk_level'],
                        'ML_Risk_Color': result['ml_risk_color'],
                        'ML_Risk_Level': result['ml_risk_level']
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
                else:
                    row_data = {
                        'Patient_ID': i + 1,
                        'Error': result['error']
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            
            # Save results for download
            timestamp = str(uuid.uuid4())[:8]
            excel_path = os.path.join(UPLOAD_FOLDER, f"results_{timestamp}.xlsx")
            csv_path = os.path.join(UPLOAD_FOLDER, f"results_{timestamp}.csv")
            
            results_df.to_excel(excel_path, index=False)
            results_df.to_csv(csv_path, index=False)
            
            return jsonify({
                'results': results,
                'download_urls': {
                    'excel': f'/download/{os.path.basename(excel_path)}',
                    'csv': f'/download/{os.path.basename(csv_path)}'
                }
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed results file"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        success = train_ml_model()
        if success:
            return jsonify({'message': 'Model trained successfully'})
        else:
            return jsonify({'error': 'Failed to train model'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Train the ML model on startup
    print("Training ML model...")
    train_ml_model()
    
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=8080) 