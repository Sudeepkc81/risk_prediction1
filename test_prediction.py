#!/usr/bin/env python3
"""
Test script for the cardiovascular risk prediction system
"""

import requests
import json

def test_prediction():
    """Test the prediction endpoint with various scenarios"""
    
    base_url = "http://localhost:8080"
    
    # Test cases based on WHO ISH chart
    test_cases = [
        {
            "name": "Low Risk - Young Non-smoker",
            "data": {
                "diabetes": "No",
                "gender": "Male", 
                "smoking": "No",
                "age": 40,
                "sbp": 120,
                "cholesterol": 4
            }
        },
        {
            "name": "Moderate Risk - Older with High BP",
            "data": {
                "diabetes": "No",
                "gender": "Male",
                "smoking": "No", 
                "age": 60,
                "sbp": 160,
                "cholesterol": 6
            }
        },
        {
            "name": "High Risk - Diabetic Smoker",
            "data": {
                "diabetes": "Yes",
                "gender": "Male",
                "smoking": "Yes",
                "age": 70,
                "sbp": 180,
                "cholesterol": 8
            }
        },
        {
            "name": "Female High Risk",
            "data": {
                "diabetes": "Yes",
                "gender": "Female",
                "smoking": "Yes",
                "age": 60,
                "sbp": 160,
                "cholesterol": 7
            }
        }
    ]
    
    print("Testing Cardiovascular Risk Prediction System")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                headers={"Content-Type": "application/json"},
                json=test_case["data"]
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"WHO ISH Risk: {result['who_ish_risk_color']} ({result['who_ish_risk_level']})")
                print(f"ML Risk: {result['ml_risk_color']} ({result['ml_risk_level']})")
                print("Parameters:")
                for key, value in result['parameters'].items():
                    print(f"  {key}: {value}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server. Make sure the application is running.")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_prediction() 