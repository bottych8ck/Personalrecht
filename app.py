import streamlit as st
import json
import time
import numpy as np
import os
import google.generativeai as genai
import re
from dotenv import load_dotenv
import openai
from groq import Groq

# Load environment variables
load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'

openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=openai_api_key)

groq_api_key = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=groq_api_key)

# Configure page and Gemini
st.set_page_config(page_title="Abfrage des Bundesmigrationsrechts", layout="wide")
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


# Function to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to get embedding of a text using Gemini
def get_embedding(text):
    result = genai.embed_content(model="models/text-embedding-004", content=text, output_dimensionality=768)
    embedding = result["embedding"]
    return np.array(embedding)

@st.cache_data
def load_data():
    # Load law_data
    with open('law_data.json', 'r') as f:
        law_data = json.load(f)
    
    # Load summary_embedding_data
    with open('summary_embeddings.json', 'r') as f:
        summary_embedding_data = json.load(f)
        
    return law_data, summary_embedding_data

def collect_articles_with_references(articles_to_evaluate, law_data):
    processed_article_ids = set()
    all_articles = []
    queue = articles_to_evaluate.copy()

    while queue:
        article = queue.pop(0)
        article_id = article['data']['ID']

        if article_id in processed_article_ids:
            continue

        processed_article_ids.add(article_id)
        all_articles.append(article)

        references = article['data'].get('references', [])

        for ref_id in references:
            if ref_id in processed_article_ids:
                continue

            found = False
            for law_articles in law_data.values():
                for ref_article_heading, ref_article_data in law_articles.items():
                    if ref_article_data['ID'] == ref_id:
                        queue.append({
                            'heading': ref_article_heading,
                            'data': ref_article_data
                        })
                        found = True
                        break
                if found:
                    break

    return all_articles

def generate_prompt(user_query, relevance, top_articles, law_data, top_knowledge_items):
    # You can implement a prompt generation function based on your requirements
    # Here we just concatenate some information as an example
    return f"Frage des Nutzers: {user_query}\n\nRelevante Artikel:\n{top_articles}\n\nZusätzliche Informationen:\n{top_knowledge_items}"

def generate_ai_response(client, prompt, model=None):
    try:
        if isinstance(client, openai.OpenAI):
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
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

def create_tooltip_html(title, content):
    return f"""
    <div class="tooltip-container">
        <span class="tooltip-text">{title}</span>
        <div class="tooltip-content">
            {content}
        </div>
    </div>
    """

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
        max-height: 400px;
        overflow-y: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
        border: 1px solid #ddd;
        scrollbar-width: thin;
        scrollbar-color: #888 #f1f1f1;
    }

    .tooltip-container:hover .tooltip-content {
        visibility: visible;
    }

    .tooltip-content::-webkit-scrollbar {
        width: 8px;
    }

    .tooltip-content::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    .tooltip-content::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    .tooltip-content::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    .select-button {
        margin-left: 10px;
    }
    </style>
    """


def main():
    st.image(logo_path, width=400)
    st.subheader("Abfrage des Migrationsrechts des Bundes")

    # Inject tooltip CSS
    st.markdown(create_tooltip_css(), unsafe_allow_html=True)

    # Initialize session state
    if 'analyzed_articles' not in st.session_state:
        st.session_state.analyzed_articles = None
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""
    if 'top_chapters' not in st.session_state:
        st.session_state.top_chapters = []

    try:
        # Load data
        law_data, summary_embedding_data = load_data()

        # Prepare chapter embeddings
        chapter_embeddings = []
        for law_full_name, sections in summary_embedding_data.items():
            for section_title, data in sections.items():
                embedding = np.array(data['embedding'])
                chapter_embeddings.append({
                    'law_full_name': law_full_name,
                    'section_title': section_title,
                    'embedding': embedding
                })

        # Create articles mapping
        articles_by_law_and_section = {}
        for law_full_name, articles in law_data.items():
            articles_by_section = {}
            for article_heading, article_data in articles.items():
                section = article_data.get('Section', None)
                if section is None:
                    continue
                if section not in articles_by_section:
                    articles_by_section[section] = []
                articles_by_section[section].append({
                    'heading': article_heading,
                    'data': article_data
                })
            articles_by_law_and_section[law_full_name] = articles_by_section

        # Query input and analyze button side by side
        query_text = st.text_input("Geben Sie Ihre rechtliche Frage ein:", key="query_input")
        analyze_button = st.button("Analysieren")

        # Main results container
        results_container = st.container()

        if analyze_button and query_text:
            st.session_state.query_text = query_text
            
            with st.spinner("Analysiere..."):
                # Semantic Search
                query_embedding = get_embedding(query_text)
                similarities = []
                for chapter in chapter_embeddings:
                    sim = cosine_similarity(query_embedding, chapter['embedding'])
                    similarities.append({
                        'law_full_name': chapter['law_full_name'],
                        'section_title': chapter['section_title'],
                        'similarity': sim
                    })

                similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
                top_chapters = similarities[:5]
                st.session_state.top_chapters = top_chapters

                # Get semantic search articles for all top chapters
                all_articles = []
                for top_chapter in top_chapters:
                    law_full_name = top_chapter['law_full_name']
                    section_title = top_chapter['section_title']
                    articles_in_section = articles_by_law_and_section.get(law_full_name, {}).get(section_title, [])
                    articles_with_refs = collect_articles_with_references(articles_in_section, law_data)
                    all_articles.extend(articles_with_refs)
                
                st.session_state.analyzed_articles = all_articles

        # Always show results if they exist in session state

        if st.session_state.top_chapters:
            with results_container:
                st.subheader("Relevante Kapitel und Artikel:")
                for idx, chapter in enumerate(st.session_state.top_chapters, 1):
                    with st.expander(f"{idx}. {chapter['law_full_name']} - {chapter['section_title']} (Relevanz: {chapter['similarity']:.2f})"):
                        law_full_name = chapter['law_full_name']
                        section_title = chapter['section_title']
                        articles_in_section = articles_by_law_and_section.get(law_full_name, {}).get(section_title, [])
                        
                        for article in articles_in_section:
                            article_id = article['data']['ID']
                            article_url = article['data'].get('URL', '#')
                            content = article['data']['content']
                            
                            # Create columns for URL and tooltip
                            col1, col2 = st.columns([10, 10])
                            
                            with col1:
                                # Display the URL link
                                st.markdown(f"[{article_id}]({article_url})")
                            
                            with col2:
                                # Display tooltip with content
                                st.markdown(create_tooltip_html(
                                    "ℹ️",  # Info emoji as hover target
                                    content
                                ), unsafe_allow_html=True)

        # AI Model section
        if st.session_state.analyzed_articles:
            st.markdown("---")
            with st.expander("🤖 Mit Sprachmodell beantworten", expanded=False):
                ai_provider = st.radio(
                    "Wählen Sie ein Sprachmodell:",
                    ("Groq Llama 3.1 (Gratis)", "OpenAI GPT-4"),
                    horizontal=True,
                    key='ai_provider'
                )
                
                current_prompt = generate_prompt(
                    st.session_state.query_text, 
                    None,
                    st.session_state.analyzed_articles, 
                    law_data, 
                    None
                )
                
                if st.button("Antwort generieren"):
                    with st.spinner('Generiere Antwort...'):
                        client = openai_client if ai_provider == "OpenAI GPT-4" else groq_client
                        response, model = generate_ai_response(client, current_prompt)
                        
                        if response:
                            st.session_state['last_answer'] = response
                            st.session_state['last_model'] = model

                if 'last_answer' in st.session_state and st.session_state['last_answer']:
                    st.success(f"Antwort erfolgreich generiert mit {st.session_state['last_model']}")
                    st.subheader(f"Antwort SubSumary ({st.session_state['last_model']}):")
                    st.markdown(st.session_state['last_answer'])

    except Exception as e:
        st.error(f"Fehler: {str(e)}")

if __name__ == "__main__":
    main()
