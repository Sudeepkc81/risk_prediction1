import csv
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.parse

class RiskPredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Read and serve the HTML template
            try:
                with open('templates/index.html', 'r', encoding='utf-8') as f:
                    content = f.read()
                self.wfile.write(content.encode('utf-8'))
            except Exception as e:
                error_html = f"""
                <html>
                <head><title>Risk Prediction</title></head>
                <body>
                    <h1>Cardiovascular Risk Prediction</h1>
                    <p>Error loading template: {e}</p>
                    <p>Please ensure templates/index.html exists.</p>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Extract parameters
                diabetes = data.get('diabetes')
                gender = data.get('gender')
                smoking = data.get('smoking')
                age = int(data.get('age'))
                sbp = int(data.get('sbp'))
                cholesterol = int(data.get('cholesterol'))
                
                # Validate inputs
                if not all([diabetes, gender, smoking, age, sbp, cholesterol]):
                    response = {'error': 'All parameters are required'}
                elif age not in [40, 50, 60, 70]:
                    response = {'error': 'Age must be 40, 50, 60, or 70'}
                elif sbp not in [120, 140, 160, 180]:
                    response = {'error': 'Systolic Blood Pressure must be 120, 140, 160, or 180'}
                elif cholesterol not in [4, 5, 6, 7, 8]:
                    response = {'error': 'Cholesterol must be 4, 5, 6, 7, or 8'}
                else:
                    # Get prediction
                    risk_color, risk_level = self.predict_risk(diabetes, gender, smoking, age, sbp, cholesterol)
                    
                    response = {
                        'who_ish_risk_color': risk_color,
                        'who_ish_risk_level': risk_level,
                        'ml_risk_color': "N/A",
                        'ml_risk_level': "ML Model not available",
                        'parameters': {
                            'diabetes': diabetes,
                            'gender': gender,
                            'smoking': smoking,
                            'age': age,
                            'sbp': sbp,
                            'cholesterol': cholesterol
                        }
                    }
                
            except Exception as e:
                response = {'error': str(e)}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def predict_risk(self, diabetes, gender, smoking, age, sbp, cholesterol):
        """Predict cardiovascular risk based on WHO ISH chart"""
        try:
            with open('WHO ish chart.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if (row['Diabetes'] == diabetes and
                        row['Gender'] == gender and
                        row['Smoking'] == smoking and
                        int(row['Age']) == age and
                        int(row['SBP']) == sbp and
                        int(row['Cholesterol']) == cholesterol):
                        risk_color = row['Risk'].title()
                        risk_level = row['Risk Level']
                        return risk_color, risk_level
            
            # If exact match not found, return default
            return "Yellow", "10-20% (Estimated)"
        except Exception as e:
            print(f"Error in predict_risk: {e}")
            return "Yellow", "10-20% (Estimated)"

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RiskPredictionHandler)
    print(f"Server running on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server() 