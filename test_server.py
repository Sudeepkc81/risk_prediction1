import urllib.request
import urllib.parse
import json

def test_server():
    try:
        # Test GET request to homepage
        print("Testing GET request to homepage...")
        response = urllib.request.urlopen('http://localhost:8080/')
        print(f"Status: {response.status}")
        print(f"Response length: {len(response.read())} bytes")
        
        # Test POST request to predict endpoint
        print("\nTesting POST request to predict endpoint...")
        data = {
            'diabetes': 'No',
            'gender': 'Male',
            'smoking': 'No',
            'age': 50,
            'sbp': 140,
            'cholesterol': 5
        }
        
        data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:8080/predict',
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        print(f"Status: {response.status}")
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Error testing server: {e}")

if __name__ == '__main__':
    test_server() 