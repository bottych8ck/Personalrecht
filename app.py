import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import requests
from groq import Groq
from streamlit.components.v1 import html
import google.generativeai as genai


# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)
    
with open('law_data.json', 'r') as file:
    law_data = json.load(file)

with open('Rechtsprechung_Embeddings.json', 'r') as file:
    Rechtsprechung_Embeddings = json.load(file)

with open('Rechtsprechung_Base.json', 'r') as file:
    Rechtsprechung_Base = json.load(file)

load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'

openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=openai_api_key)

groq_api_key = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=groq_api_key)

st.set_page_config(page_title="Abfrage des Bundesmigrationsrechts", layout="wide")
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def generate_ai_response(client, prompt, model=None):
    try:
        if isinstance(client, openai.OpenAI):
            response = client.chat.completions.create(
                model="GPT-4o:",
                messages=[
                    {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content, "GPT-4"
        else:  # Groq client
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-70b-versatile"
            )
            return response.choices[0].message.content, "Llama 3.1"
    except Exception as e:
        st.error(f"Error with {client.__class__.__name__}: {str(e)}")
        return None, None

def create_tooltip_css():
    return """
    <style>
    .tooltip-container {
        position: relative;
        display: inline-block;
        margin: 5px 0;
        width: 100%;
    }

    .tooltip-text {
        cursor: pointer;
        border-bottom: 1px dotted #666;
        display: inline-block;
        padding: 5px;
    }

    .tooltip-content {
        visibility: hidden;
        position: absolute;
        left: 0;
        background-color: white;
        color: black;
        padding: 15px;
        border-radius: 6px;
        width: 100%;
        max-height: 300px;
        overflow-y: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
        border: 1px solid #ddd;
    }

    .tooltip-container:hover .tooltip-content {
        visibility: visible;
    }

    .select-button {
        margin-left: 10px;
    }
    </style>
    """

def create_tooltip_html(title, content):
    return f"""
    <div class="tooltip-container">
        <span class="tooltip-text">{title}</span>
        <div class="tooltip-content">
            {content}
        </div>
    </div>
    """

def keyword_search(keyword, law_data, Rechtsprechung_Base):
    keyword = keyword.lower()
    
    matching_articles = {}
    for uid, article in law_data.items():
        if keyword in article.get('Title', '').lower():
            matching_articles[uid] = article
            continue
            
        if any(keyword in paragraph.lower() for paragraph in article.get('Inhalt', [])):
            matching_articles[uid] = article
            
    matching_items = {}
    for item_id, item in Rechtsprechung_Base.items():
        if keyword in item.get('Title', '').lower():
            matching_items[item_id] = item
            continue
            
        if any(keyword in content.lower() for content in item.get('Content', [])):
            matching_items[item_id] = item
            
    return matching_articles, matching_items

def update_file_in_github(file_path, content, commit_message="Update file"):
    repo_owner = os.getenv('GITHUB_REPO_OWNER')
    repo_name = os.getenv('GITHUB_REPO_NAME')
    token = os.getenv('GITHUB_TOKEN')
    branch_name = os.getenv('GITHUB_BRANCH', 'AV-SA')

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={branch_name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    sha = response.json()["sha"]

    data = {
        "message": commit_message,
        "content": base64.b64encode(content.encode('utf-8')).decode('utf-8'),
        "sha": sha,
        "branch": branch_name
    }

    response = requests.put(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def main_app():
    st.image(logo_path, width=400)
    st.subheader("Abfrage des Thurgauer Personalrechts")
    
    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200, key="user_query_text_area")
    
    if st.button("Bearbeiten"):
        st.session_state['last_question'] = user_query
    
if __name__ == "__main__":
    main_app()
