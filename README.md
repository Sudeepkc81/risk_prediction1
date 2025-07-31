# Cardiovascular Risk Prediction System

A web-based application for predicting cardiovascular risk using the WHO ISH (World Health Organization/International Society of Hypertension) chart and machine learning models.

## Features

- **WHO ISH Chart-based Prediction**: Uses the standard WHO ISH cardiovascular risk prediction chart
- **Machine Learning Model**: Additional prediction using Random Forest classifier trained on clinical data
- **6-Parameter Assessment**: 
  - Diabetes (Yes/No)
  - Gender (Male/Female)
  - Smoking Status (Yes/No)
  - Age (40, 50, 60, 70 years)
  - Systolic Blood Pressure (120, 140, 160, 180 mmHg)
  - Cholesterol (4, 5, 6, 7, 8 mmol/L)
- **Risk Levels**: 
  - Green: <10% risk
  - Yellow: 10-20% risk
  - Orange: 20-30% risk
  - Red-Orange: 30-40% risk
  - Red: >40% risk

## Files Description

- `risk_prediction_app.py`: Main Flask application
- `templates/index.html`: Web interface template
- `WHO ish chart.csv`: WHO ISH chart data for risk lookup
- `RishikaData_Cleaned_mmol.xlsx`: Clinical dataset for ML model training
- `requirements.txt`: Python dependencies
- `protocol 2.PNG`: Protocol documentation
- `risk_prediction_rules.docx`: Risk prediction rules documentation

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Ensure all data files are in the project directory**:
   - `WHO ish chart.csv`
   - `RishikaData_Cleaned_mmol.xlsx`

## Usage

1. **Start the application**:
   ```bash
   python3 risk_prediction_app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:8080
   ```

3. **Enter the 6 parameters**:
   - Select diabetes status
   - Choose gender
   - Indicate smoking status
   - Select age group
   - Enter systolic blood pressure
   - Specify cholesterol level

4. **Click "Calculate Risk"** to get predictions

5. **View results**:
   - WHO ISH Chart prediction
   - Machine Learning model prediction
   - Input parameters summary

## How It Works

### WHO ISH Chart Prediction
The application uses the WHO ISH chart data to perform exact lookups based on the 6 input parameters. If an exact match isn't found, it finds the closest match using the available data.

### Machine Learning Model
- **Training**: Uses the cleaned clinical dataset (`RishikaData_Cleaned_mmol.xlsx`)
- **Features**: Age, gender, smoking status, diabetes, systolic blood pressure, cholesterol
- **Algorithm**: Random Forest Classifier
- **Target**: Cardiovascular risk (derived from clinical outcomes or synthetic risk score)

### Risk Assessment
The system provides two types of predictions:
1. **WHO ISH Chart-based**: Standard clinical risk assessment
2. **ML Model-based**: Additional prediction using trained machine learning model

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Risk prediction endpoint
  - Input: JSON with 6 parameters
  - Output: Risk predictions and levels
- `POST /train_model`: Retrain ML model endpoint

## Technical Details

### Data Processing
- **WHO ISH Chart**: Direct lookup from CSV data
- **Clinical Data**: Preprocessed for ML training with feature encoding
- **Validation**: Input validation for all parameters

### Model Training
- **Algorithm**: Random Forest Classifier
- **Features**: 6 cardiovascular risk factors
- **Scaling**: StandardScaler for feature normalization
- **Validation**: 80/20 train-test split

### Web Interface
- **Framework**: Flask
- **Frontend**: Bootstrap 5, Font Awesome
- **Responsive**: Mobile-friendly design
- **Real-time**: AJAX-based predictions

## Risk Levels Explanation

- **Green (<10%)**: Low cardiovascular risk
- **Yellow (10-20%)**: Moderate risk, lifestyle modifications recommended
- **Orange (20-30%)**: Moderate-high risk, consider medication
- **Red-Orange (30-40%)**: High risk, medication likely needed
- **Red (>40%)**: Very high risk, immediate medical attention required

## Troubleshooting

1. **Port already in use**: Change port in `risk_prediction_app.py`
2. **Missing dependencies**: Run `pip3 install -r requirements.txt`
3. **Data file errors**: Ensure CSV and Excel files are in the project directory
4. **ML model not training**: Check data quality in Excel file

## Security Notes

- This is a development application
- No patient data is stored
- All predictions are calculated in real-time
- No persistent data storage

## Future Enhancements

- Additional risk factors
- More sophisticated ML models
- Patient data management
- Export functionality
- Mobile app version

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project documentation. 