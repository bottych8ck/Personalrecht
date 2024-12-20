from flask import Flask, request, jsonify
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Configure Gemini API for embeddings
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def load_data():
    """Load all necessary data files"""
    base_dir = 'bildungsrechtsentscheide'
    
    with open(f'{base_dir}/article_embeddings.json', 'r') as file:
        article_embeddings = json.load(file)
        
    with open(f'{base_dir}/law_data.json', 'r') as file:
        law_data = json.load(file)

    with open(f'{base_dir}/Rechtssprechung-Embeddings.json', 'r') as file:
        rechtssprechung_embeddings = json.load(file)

    with open(f'{base_dir}/Rechtssprechung-Base.json', 'r') as file:
        rechtssprechung_base = json.load(file)
        
    return article_embeddings, law_data, rechtssprechung_embeddings, rechtssprechung_base

def get_embeddings(text):
    """Generate embeddings using Gemini API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            output_dimensionality=768
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def calculate_similarities(query_vector, embeddings_dict):
    """Calculate cosine similarities between query and stored embeddings"""
    similarities = {}
    query_vector = np.array(query_vector).reshape(1, -1)
    
    for uid, data in embeddings_dict.items():
        if isinstance(data, dict) and 'embedding' in data:
            doc_vector = np.array(data['embedding']).reshape(1, -1)
        else:
            doc_vector = np.array(data).reshape(1, -1)
        
        similarity = cosine_similarity(query_vector, doc_vector)[0][0]
        similarities[uid] = similarity
    
    return similarities

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint for finding relevant articles and decisions"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        limit = data.get('limit', 10)  # Number of results to return, default 10
        
        # Load data
        article_embeddings, law_data, rechtssprechung_embeddings, rechtssprechung_base = load_data()
        
        # Get embeddings for the query
        query_vector = get_embeddings(query)
        if not query_vector:
            return jsonify({'error': 'Failed to generate embeddings'}), 500
            
        # Find relevant articles and decisions
        article_similarities = calculate_similarities(query_vector, article_embeddings)
        decision_similarities = calculate_similarities(query_vector, rechtssprechung_embeddings)
        
        # Get top results
        top_articles = sorted(article_similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
        top_decisions = sorted(decision_similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Format response
        result = {
            'articles': [
                {
                    'id': uid,
                    'title': law_data.get(str(uid), {}).get('Title', ''),
                    'content': law_data.get(str(uid), {}).get('Inhalt', []),
                    'law_name': law_data.get(str(uid), {}).get('Name', [''])[0],
                    'relevance_score': float(score)
                }
                for uid, score in top_articles
            ],
            'decisions': [
                {
                    'id': item_id,
                    'name': rechtssprechung_base.get(item_id, {}).get('name', ''),
                    'summary': rechtssprechung_base.get(item_id, {}).get('summary', {}),
                    'relevance_score': float(score)
                }
                for item_id, score in top_decisions
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080))) 