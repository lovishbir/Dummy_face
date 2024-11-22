
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from logic import analyze_image  # Assuming this is the main function from your notebook logic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = analyze_image(filepath)  # Call your notebook function
        os.remove(filepath)  # Clean up uploaded file
        return result  # Return result to the user

if __name__ == '__main__':
    app.run(debug=True)
