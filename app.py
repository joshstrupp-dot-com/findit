# Import necessary libraries for web app, data processing, and text analysis
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import boto3
import io

# Initialize Flask web application
app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')

# Step 1: Load the review data from S3 CSV file
print("Loading data from S3...")
bucket_name = 'findit-selfhelp'
file_name = 'sentiment_analysis_results.csv'

try:
    response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    csv_content = response['Body'].read()
    df = pd.read_csv(io.BytesIO(csv_content))
    print("Successfully loaded data from S3")
except Exception as e:
    print(f"Error loading data from S3: {str(e)}")
    raise

# Step 2: Clean the data to handle missing values
print("Cleaning data...")
# Remove any reviews that don't have text content
df = df.dropna(subset=['review_text'])
# Replace missing values in other columns with empty strings or zeros
df = df.fillna({'name': '', 'author_clean': '', 'star_rating': 0})

# Step 3: Create a search index using TF-IDF (Term Frequency-Inverse Document Frequency)
print("Creating search index...")
# Initialize the vectorizer with English stop words removed
vectorizer = TfidfVectorizer(stop_words='english')
# Convert all review texts into numerical vectors
tfidf_matrix = vectorizer.fit_transform(df['review_text'])

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the search endpoint that accepts POST requests
@app.route('/search', methods=['POST'])
def search():
    # Get the search query from the request
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Step 4: Convert the search query into the same numerical format as reviews
    query_vector = vectorizer.transform([query])
    
    # Step 5: Calculate how similar the query is to each review
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Step 6: Find the 5 most similar reviews
    top_indices = similarities.argsort()[-5:][::-1]
    
    # Step 7: Format the results
    results = []
    for idx in top_indices:
        results.append({
            'name': df.iloc[idx]['name'],
            'author': df.iloc[idx]['author_clean'],
            'rating': float(df.iloc[idx]['star_rating']),
            'review': df.iloc[idx]['review_text'],
            'similarity': float(similarities[idx])
        })
    
    # Return the results as JSON
    return jsonify(results)

# Run the application when this file is executed directly
if __name__ == '__main__':
    app.run(debug=True, port=5001) 