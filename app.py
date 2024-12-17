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

# Set page config ONCE at the very top
st.set_page_config(page_title="Abfrage des kantonalen Bildungsrechts", layout="wide")

# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)
    
with open('law_data.json', 'r') as file:
    law_data = json.load(file)

with open('Rechtssprechung-Embeddings.json', 'r') as file:
    Rechtssprechung_Embeddings = json.load(file)

with open('Rechtssprechung-Base.json', 'r') as file:
    Rechtssprechung_Base = json.load(file)

load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'

openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=openai_api_key)

groq_api_key = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=groq_api_key)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def get_embeddings(text):
    """Generate embeddings for a given text using Gemini API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            output_dimensionality=768  # Match the dimension used in processor
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
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

def generate_ai_response(client, prompt, model=None):
    try:
        if isinstance(client, openai.OpenAI):
            response = client.chat.completions.create(
                model="gpt-4o:",
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

def keyword_search(keyword, law_data, Rechtssprechung_Base):
    keyword = keyword.lower()
    
    matching_articles = {}
    for uid, article in law_data.items():
        if keyword in article.get('Title', '').lower():
            matching_articles[uid] = article
            continue
            
        if any(keyword in paragraph.lower() for paragraph in article.get('Inhalt', [])):
            matching_articles[uid] = article
            
    matching_items = {}
    for item_id, item in Rechtssprechung_Base.items():
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

def get_article_content(uid, law_data):
    """Extract article content from law data"""
    article = law_data.get(uid, {})
    
    # Use the article ID as title since it contains the article name
    title = uid  # e.g., "1 - Volksschule"
    paragraphs = article.get('Inhalt', [])
    law_name = article.get('Name', ['Unknown Law'])[0] if article.get('Name') else 'Unknown Law'
    law_url = article.get('URL', None)
    
    return title, paragraphs, law_name, law_url

def generate_prompt(question, top_articles, law_data, top_knowledge_items):
    """Generate a prompt combining the question with relevant articles and decisions"""
    prompt = f"Frage: {question}\n\nRelevante Gesetzesartikel:\n"
    
    # Add relevant articles
    for uid, _ in top_articles:
        article = law_data.get(str(uid), {})
        title = uid
        content = '\n'.join(article.get('Inhalt', []))
        law_name = article.get('Name', [''])[0]
        prompt += f"\n{title} - {law_name}:\n{content}\n"
    
    # Add only top 10 relevant decisions
    prompt += "\nRelevante Entscheide:\n"
    for item_id, _ in top_knowledge_items[:10]:  # Limit to first 10 decisions
        item = Rechtssprechung_Base.get(item_id, {})
        summary = item.get('summary', {})
        if summary:
            prompt += f"\nEntscheid {item.get('name', 'Unbekannt')}:\n"
            prompt += f"Sachverhalt: {summary.get('Sachverhalt', '')}\n"
            prompt += f"Erwägungen: {summary.get('Erwägungen', '')}\n"
            prompt += f"Entscheid: {summary.get('Entscheid', '')}\n"
    
    prompt += "\nBitte beantworte die Frage basierend auf den oben genannten Gesetzesartikeln und Entscheiden."
    return prompt

def main_app():
    st.image(logo_path, width=400)
    st.subheader("Abfrage des kantonalen Bildungsrechts und bildungsrechtlicher Entscheide")

    # Initialize session state
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'last_answer_gpt4o' not in st.session_state:
        st.session_state['last_answer_gpt4o'] = None
    if 'top_articles' not in st.session_state:
        st.session_state['top_articles'] = []
    if 'top_knowledge_items' not in st.session_state:
        st.session_state['top_knowledge_items'] = []
    if 'generated_prompt' not in st.session_state:
        st.session_state['generated_prompt'] = ""
    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False
    if 'show_form' not in st.session_state:
        st.session_state['show_form'] = False
    if 'delete_form' not in st.session_state:
        st.session_state['delete_form'] = False
    if 'generating_answer' not in st.session_state:
        st.session_state.generating_answer = False
    if 'start_generating_answer' not in st.session_state:
        st.session_state.start_generating_answer = False
    if 'last_model' not in st.session_state:
        st.session_state.last_model = False

    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200, key="user_query_text_area")

    # On pressing "Bearbeiten"
    if st.button("Bearbeiten"):
        st.session_state['last_question'] = user_query
        st.session_state['last_answer'] = None
        st.session_state.generating_answer = False

        # Get embeddings for the query and find relevant articles
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.session_state.top_articles = sorted_articles[:10]

        # Find relevant Rechtssprechung items
        knowledge_similarities = calculate_similarities(query_vector, Rechtssprechung_Embeddings)
        st.session_state.top_knowledge_items = [
            (item_id, score) for item_id, score in sorted(knowledge_similarities.items(), key=lambda x: x[1], reverse=True)
        ]

        st.session_state.submitted = True

    if st.session_state.get('submitted'):
        with st.expander("Am besten auf die Anfrage passende Bestimmungen und Wissenselemente", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Bestimmungen")
                for uid, score in st.session_state.top_articles:
                    article_info = law_data.get(str(uid), None)
                    if article_info:
                        title, all_paragraphs, law_name, law_url = get_article_content(str(uid), law_data)
                        law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                        if law_url:
                            law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"

                        title_clean = title.strip('*')
                        st.markdown(f"**{title_clean} - {law_name_display}**", unsafe_allow_html=True)

                        if all_paragraphs:
                            for paragraph in all_paragraphs:
                                st.write(paragraph)
                        else:
                            st.write("Kein Inhalt verfügbar.")

            with col2:
                st.markdown("#### Wissenselemente")
                
                # Get all decisions
                decisions = st.session_state.top_knowledge_items
                
                # Show first 10 by default
                for item_id, _ in decisions[:10]:
                    item = Rechtssprechung_Base.get(item_id, {})
                    name = item.get('name', 'Unbekannt')
                    source_url = item.get('source_url', '')
                    
                    if source_url:
                        title = f"**Entscheid: <a href='{source_url}' target='_blank'>{name}</a>**"
                    else:
                        title = f"**Entscheid: {name}**"
                    
                    summary = item.get('summary', {})
                    st.markdown(title, unsafe_allow_html=True)
                    
                    if summary:
                        if 'Sachverhalt' in summary:
                            st.markdown("**Sachverhalt:**")
                            st.write(summary['Sachverhalt'])
                        if 'Erwägungen' in summary:
                            st.markdown("**Erwägungen:**")
                            st.write(summary['Erwägungen'])
                        if 'Entscheid' in summary:
                            st.markdown("**Entscheid:**")
                            st.write(summary['Entscheid'])
                    
                    st.markdown("---")
                
                # Show "Show More" button if there are more than 10 decisions
                if len(decisions) > 10:
                    if st.button("Weitere Entscheide anzeigen"):
                        for item_id, _ in decisions[10:]:
                            item = Rechtssprechung_Base.get(item_id, {})
                            name = item.get('name', 'Unbekannt')
                            source_url = item.get('source_url', '')
                            
                            if source_url:
                                title = f"**Entscheid: <a href='{source_url}' target='_blank'>{name}</a>**"
                            else:
                                title = f"**Entscheid: {name}**"
                            
                            summary = item.get('summary', {})
                            st.markdown(title, unsafe_allow_html=True)
                            
                            if summary:
                                if 'Sachverhalt' in summary:
                                    st.markdown("**Sachverhalt:**")
                                    st.write(summary['Sachverhalt'])
                                if 'Erwägungen' in summary:
                                    st.markdown("**Erwägungen:**")
                                    st.write(summary['Erwägungen'])
                                if 'Entscheid' in summary:
                                    st.markdown("**Entscheid:**")
                                    st.write(summary['Entscheid'])
                            
                            st.markdown("---")

    if st.session_state.get('submitted'):
        st.markdown("---")
        with st.expander("🔍 Zusätzliche Stichwortsuche", expanded=False):
            st.write(create_tooltip_css(), unsafe_allow_html=True)
            keyword = st.text_input("Stichwort eingeben und Enter drücken:")
            st.markdown("Auswählen, welche Artikel oder Wissenselemente für die Antwort zusätzlich berücksichtigt werden sollen:")
            if keyword:
                matching_articles, matching_items = keyword_search(keyword, law_data, Rechtssprechung_Base)

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
                            if st.checkbox("", key=f"select_article_{uid}"):
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

                    if selected_article_uids and st.button("Ausgewählte Artikel hinzufügen"):
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
                            if st.checkbox("", key=f"select_item_{item_id}"):
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

                    if selected_item_ids and st.button("Ausgewählte Wissenselemente hinzufügen"):
                        existing_ids = [item_id for item_id, _ in st.session_state.top_knowledge_items]
                        for item_id in selected_item_ids:
                            if item_id not in existing_ids:
                                st.session_state.top_knowledge_items.append((item_id, 1.0))
                        st.success("Ausgewählte Wissenselemente wurden zu den relevanten Wissenselementen hinzugefügt")


        with st.expander("Neues Wissen hinzufügen", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Neues Wissenselement hinzufügen"):
                    st.session_state.show_form = not st.session_state.show_form

                if st.session_state.show_form:
                    with st.form(key='add_knowledge_form'):
                        title = st.text_input("Titel", value=f"Hinweis zu folgender Frage: {st.session_state['last_question']}")
                        content = st.text_area("Inhalt")
                        category = "User-Hinweis"
                        selected_german_tags = st.multiselect(
                            "Anwendbarkeit: Auf welche Personalkategorie ist das neue Wissen anwendbar? Bitte auswählen, mehrfache Auswahl ist erlaubt.",
                            list(set(tags_mapping.values())),
                            default=[
                                "Staatspersonal",
                                "Lehrperson VS",
                                "Lehrperson Sek II"
                            ]
                        )
                        submit_button = st.form_submit_button(label='Hinzufügen')

                        if submit_button and title and content:
                            # Convert the selected German tags to their corresponding English tags
                            selected_english_tags = []
                            for selected_german_tag in selected_german_tags:
                                selected_english_tags.extend(reverse_tags_mapping[selected_german_tag])
                            # Adjust this function as needed if it's meant to update Rechtssprechung_Base
                            add_to_knowledge_base(title, content, category, selected_english_tags)
                            st.success("Neues Wissen erfolgreich hinzugefügt!")

                if 'delete_form' not in st.session_state:
                    st.session_state.delete_form = False

            with col2:
                if st.button("Wissenselement löschen"):
                    st.session_state.delete_form = not st.session_state.delete_form

                if st.session_state.delete_form:
                    with st.form(key='delete_knowledge_form'):
                        entry_id_to_delete = st.selectbox(
                            "Wählen Sie das Wissenselement zum Löschen aus:", 
                            list(Rechtssprechung_Base.keys()),
                            format_func=lambda x: f"{x}: {Rechtssprechung_Base[x]['Title']}"
                        )
                        delete_button = st.form_submit_button(label='Löschen')

                        if delete_button and entry_id_to_delete:
                            # Adjust this function as needed if it's meant to delete from Rechtssprechung_Base
                            delete_from_knowledge_base(entry_id_to_delete)

        st.write("")
        st.write("")

        # genAI-Teil
        with st.expander("🤖 Mit Sprachmodell beantworten", expanded=True):
            ai_provider = st.radio(
                "Wählen Sie ein Sprachmodell:",
                ("Groq Llama 3.1 (Gratis)", "OpenAI gpt-4o"),
                horizontal=True,
                key='ai_provider'
            )

            # Generate fresh prompt
            current_prompt = generate_prompt(
                st.session_state['last_question'], 
                st.session_state.top_articles, 
                law_data, 
                st.session_state.top_knowledge_items
            )

            if st.button("Antwort generieren"):
                with st.spinner('Generiere Antwort...'):
                    client = openai_client if ai_provider == "OpenAI GPT-4" else groq_client
                    response, model = generate_ai_response(client, current_prompt)

                    if response:
                        st.session_state['last_answer'] = response
                        st.session_state['last_model'] = model

            # Display answer
            answer_container = st.container()
            with answer_container:
                if 'last_answer' in st.session_state and st.session_state['last_answer']:
                    st.success(f"Antwort erfolgreich generiert mit {st.session_state['last_model']}")
                    st.subheader(f"Antwort SubSumary ({st.session_state['last_model']}):")
                    st.markdown(st.session_state['last_answer'])
                    html(generate_html_with_js(st.session_state['last_answer']))

    
if __name__ == "__main__":
    main_app()
