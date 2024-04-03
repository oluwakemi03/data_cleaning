import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import os
from scipy.signal import medfilt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_missing_data(df):
    # Replace 'NA' strings with NaN (missing value indicator)
    df.replace('NA', pd.NA, inplace=True)
    
    # Fill missing values in all columns with NaN
    df.fillna(pd.NA, inplace=True)

    return df

def preprocess_data(df):
    # Iterate through columns and check if they are string type
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove semicolons and any other trailing characters from string columns
            df[col] = df[col].str.rstrip(';')
    return df


def median_filtering(df, window_size=3):
    # Apply median filtering to numerical columns
    numerical_cols = df.select_dtypes(include='number').columns
    df[numerical_cols] = df[numerical_cols].apply(lambda x: medfilt(x, kernel_size=window_size))
    return df

def detect_outliers_iqr(df):
    # Ensure the last column is converted to string
    df.iloc[:, -1] = df.iloc[:, -1].astype(str)
    
    # Remove the last semicolon from the last column
    df.iloc[:, -1] = df.iloc[:, -1].str.rstrip(';')
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    # Calculate the first and third quartiles
    for col in numeric_df:
        # Calculate quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define upper and lower bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Remove outliers
        df.loc[outliers, col] = pd.NA
    
    return numeric_df




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
            df = handle_missing_data(df)
            # Preprocess the data
            df = preprocess_data(df)
            # Apply median filtering
            df = median_filtering(df, window_size=3)
            # Detect and remove outliers using IQR method
            df = detect_outliers_iqr(df)
            # Save the cleaned CSV file
            cleaned_filename = filename.split('.')[0] + '_cleaned.csv'
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename), index=False)
            
            # Calculate number of missing values
            num_missing_values = df.isnull().sum().sum()

            # Pass the number of missing values to the template
            return render_template('index.html', num_missing_values=num_missing_values)

@app.route('/uploads/<filename>')
def download_file(filename):
    return render_template('download.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
