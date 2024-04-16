import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
from scipy.signal import medfilt
import xml.etree.ElementTree as ET


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xml'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_missing_data(df):
    # Replace 'NA' strings with NaN (missing value indicator)
    df.replace('NA', pd.NA, inplace=True)
    
    # Fill missing values in numeric columns with the median of each column
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Calculate number of missing values
    num_missing_values = df.isnull().sum().sum()
    
    return df, num_missing_values

def detect_outliers_iqr(df):
    # Drop the last column
    df = df.iloc[:, :-1]
    
    # Calculate the first and third quartiles for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define upper and lower bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Identify outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Remove outliers
        df.loc[outliers, col] = pd.NA
    
    return df

def median_filtering(df, window_size=3):
    # Apply median filtering to numerical columns after filling missing values
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        # Fill missing values in each column with the median of that column
        df[col].fillna(df[col].median(), inplace=True)
        
        # Apply median filter
        df[col] = medfilt(df[col], kernel_size=window_size)
    
    return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            # Check if the file is XML
            if filename.endswith('.xml'):
                print("File exist")
                tree = ET.parse(file)
                root = tree.getroot()
                
                # Extract data from XML
                bpm_list = []
                time_list = []
                for bpm_elem in root.findall('.//InstantaneousBeatsPerMinute'):
                    bpm = int(bpm_elem.attrib['bpm'])
                    time = bpm_elem.attrib['time']
                    bpm_list.append(bpm)
                    time_list.append(time)

                # Create DataFrame
                data = {'bpm': bpm_list, 'time': time_list}
                df = pd.DataFrame(data)

                # Write DataFrame to Excel
                cleaned_filename = filename.split('.')[0] + '_cleaned.xlsx'
                df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename), index=False)
                return render_template('index.html', excel_generated=True, cleaned_filename=cleaned_filename)
            else:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Convert TXT to CSV if uploaded file is TXT
                if filename.endswith('.txt'):
                    converted_filename = filename[:-4] + '_converted.csv'
                    # Read the TXT file with commas as delimiter
                    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), delimiter='[;,]', header=None)
                    # Save the DataFrame as CSV with commas as delimiter
                    df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], converted_filename), index=False, header=False, sep=',')
                    filename = converted_filename
                # Read the CSV file
                df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Handle missing data
                df, num_missing_values = handle_missing_data(df)
                # Detect outliers
                df = detect_outliers_iqr(df)
                # Apply median filtering
                df = median_filtering(df)
                
                cleaned_filename = filename.split('.')[0] + '_cleaned.csv'
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename), index=False)
                
                # Pass the number of missing values to the template
                return render_template('index.html', num_missing_values=num_missing_values, cleaned_filename=cleaned_filename)


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
