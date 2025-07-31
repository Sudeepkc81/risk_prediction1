import pandas as pd
import os

def create_sample_template():
    """Create a sample Excel template for batch uploads"""
    
    # Sample data with correct format
    sample_data = {
        'Diabetes': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Smoking': ['No', 'Yes', 'No', 'No', 'Yes'],
        'Age': [60, 50, 70, 40, 60],
        'SBP': [160, 140, 180, 120, 160],
        'Cholesterol': [6, 5, 7, 4, 8]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save as Excel file
    template_path = 'sample_batch_template.xlsx'
    df.to_excel(template_path, index=False)
    
    print(f"Sample template created: {template_path}")
    print("\nTemplate includes:")
    print("- 5 sample patients with valid data")
    print("- All required columns: Diabetes, Gender, Smoking, Age, SBP, Cholesterol")
    print("- Valid values for each parameter")
    print("\nValid values:")
    print("- Diabetes: Yes, No")
    print("- Gender: Male, Female")
    print("- Smoking: Yes, No")
    print("- Age: 40, 50, 60, 70")
    print("- SBP: 120, 140, 160, 180")
    print("- Cholesterol: 4, 5, 6, 7, 8")

if __name__ == "__main__":
    create_sample_template() 