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

def keyword_search(keyword, law_data, knowledge_base):
    keyword = keyword.lower()
    
    matching_articles = {}
    for law_name, articles in law_data.items():
        for article_title, article_data in articles.items():
            if keyword in article_title.lower():
                matching_articles[article_data['ID']] = {
                    'Title': article_title,
                    'Name': law_name,
                    'Inhalt': [article_data['content']]
                }
                continue
                
            if keyword in article_data['content'].lower():
                matching_articles[article_data['ID']] = {
                    'Title': article_title,
                    'Name': law_name, 
                    'Inhalt': [article_data['content']]
                }
            
    matching_items = {}
    for item_id, item in knowledge_base.items():
        if keyword in item.get('Title', '').lower():
            matching_items[item_id] = item
            continue
            
        if any(keyword in content.lower() for content in item.get('Content', [])):
            matching_items[item_id] = item
            
    return matching_articles, matching_items

            
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
        
    # Load knowledge_base
    with open('knowledge_base.json', 'r') as f:
        knowledge_base = json.load(f)
        
    return law_data, summary_embedding_data, knowledge_base

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
    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = None
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
        law_data, summary_embedding_data, knowledge_base = load_data()

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

        # Query input and analyze button
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
        if st.session_state.get('top_chapters'):
            st.markdown("---")

        with st.expander("🔍 Zusätzliche Stichwortsuche", expanded=False):
            st.write(create_tooltip_css(), unsafe_allow_html=True)
            # Add a key to the text_input to force refresh
            keyword = st.text_input("Stichwort eingeben und Enter drücken:", key=f"keyword_search_{time.time()}")
            st.markdown("Auswählen, welche Artikel oder Wissenselemente für die Antwort zusätzlich berücksichtigt werden sollen:")
            if keyword:
                matching_articles, matching_items = keyword_search(keyword, law_data, knowledge_base)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Gefundene Gesetzesartikel")
                    selected_article_uids = []
                    
                    for uid, article in matching_articles.items():
                        title = article.get('Title', 'Unknown Title')
                        law_name = article.get('Name', 'Unbekanntes Gesetz')
                        content = '<br>'.join(article.get('Inhalt', []))
                        
                        col_select, col_content = st.columns([1, 4])
                        with col_select:
                            # Add unique key using keyword and uid
                            if st.checkbox("", key=f"select_article_{keyword}_{uid}"):
                                selected_article_uids.append(uid)
                        with col_content:
                            st.write(
                                create_tooltip_html(
                                    f"{title} - {law_name}", 
                                    content
                                ), 
                                unsafe_allow_html=True
                            )
                        st.markdown("---")
                    
                    # Add keyword to button key
                    if selected_article_uids and st.button("Ausgewählte Artikel hinzufügen", key=f"add_articles_{keyword}"):
                        existing_uids = [uid for uid, _ in st.session_state.top_articles]
                        for uid in selected_article_uids:
                            if uid not in existing_uids:
                                st.session_state.top_articles.append((uid, 1.0))
                        st.success("Ausgewählte Artikel wurden zu den relevanten Artikeln hinzugefügt")
    
                with col2:
                    st.markdown("#### Gefundene Wissenselemente")
                    selected_item_ids = []
                    
                    for item_id, item in matching_items.items():
                        title = item.get('Title', 'Unknown Title')
                        content = ' '.join(item.get('Content', []))
                        
                        col_select, col_content = st.columns([1, 4])
                        with col_select:
                            # Add unique key using keyword and item_id
                            if st.checkbox("", key=f"select_item_{keyword}_{item_id}"):
                                selected_item_ids.append(item_id)
                        with col_content:
                            st.write(
                                create_tooltip_html(
                                    title,
                                    content
                                ),
                                unsafe_allow_html=True
                            )
                        st.markdown("---")
                    
                    # Add keyword to button key
                    if selected_item_ids and st.button("Ausgewählte Wissenselemente hinzufügen", key=f"add_items_{keyword}"):
                        existing_ids = [item_id for item_id, _ in st.session_state.top_knowledge_items]
                        for item_id in selected_item_ids:
                            if item_id not in existing_ids:
                                st.session_state.top_knowledge_items.append((item_id, 1.0))
                        st.success("Ausgewählte Wissenselemente wurden zu den relevanten Wissenselementen hinzugefügt")

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
