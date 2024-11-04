import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import spacy
from pathlib import Path

# Create a models directory in the current working directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Set the spacy model path environment variable
os.environ["SPACY_MODEL_PATH"] = str(MODEL_DIR)

def load_spacy_model():
    model_name = "de_core_news_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Download to custom directory
        spacy.cli.download(model_name, str(MODEL_DIR))
        nlp = spacy.load(MODEL_DIR / model_name)
    return nlp

# Load model
nlp = load_spacy_model()

import openai
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from rank_bm25 import BM25Okapi
import pickle
import re
import nltk
from nltk.stem.snowball import GermanStemmer
from typing import List, Optional
from groq import Groq
import spacy
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

# Load German spaCy model
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    spacy.cli.download("de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)

load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'
openAI_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Set up clients for OpenAI and Groq
openai_client = openai.OpenAI(api_key=openAI_api_key)
groq_client = Groq(api_key=groq_api_key)

def get_embeddings(text):
    res = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def get_relevant_articles(law_data, relevance):
    relevant_articles = {}
    for section, section_data in law_data.items():
        if is_relevant_article(section_data, relevance):
            relevant_articles[section] = section_data
    return relevant_articles

def calculate_similarities(query_vector, article_embeddings):
    query_vector = np.array(query_vector).reshape(1, -1)
    similarities = {}
    
    for title, article_vector in article_embeddings.items():
        try:
            article_vector = np.asarray(article_vector, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(query_vector, article_vector)[0][0]
            similarities[title] = similarity
        except TypeError as e:
            print(f"Error processing article '{title}': {e}")
    return similarities

def get_article_content(title, law_data):
    section_data = law_data.get(title, {})
    all_paragraphs = section_data.get('Inhalt', [])
    law_name = section_data.get("Name", "Unbekanntes Gesetz")
    law_url = section_data.get("URL", "")

    mentioned_articles = section_data.get("Im § erwähnter Artikel des EOG", [])
    if mentioned_articles:
        all_paragraphs += ["Im § erwähnter Artikel des EOG:"] + mentioned_articles

    return (title, all_paragraphs, law_name, law_url)

def generate_html_with_js(prompt):
    return f"""
    <textarea id='text_area' style='opacity: 0; position: absolute; left: -9999px;'>{prompt}</textarea>
    <script>
    function copyToClipboard() {{
        var copyText = document.getElementById('text_area');
        copyText.style.opacity = 1; // Make the textarea visible to enable selection
        copyText.select();
        navigator.clipboard.writeText(copyText.value).then(function() {{
            alert('Copied to clipboard!');
        }}, function(err) {{
            console.error('Could not copy text: ', err);
        }});
        copyText.style.opacity = 0; // Hide the textarea again
    }}
    copyToClipboard();
    </script>
    """

def generate_prompt(user_query, top_articles, law_data):
    prompt = f"Beantworte folgende Frage: \"{user_query}\"\n\n"
    prompt += "Beantworte die Frage nur gestützt auf einen oder mehrere der folgenden §. Prüfe zuerst, ob der § überhaupt auf die Frage anwendbar ist. Nenne unbedingt immer den § und das dazugehörige Gesetz oder die zugehörige Verordnung. Wenn er nicht anwendbar ist, vergiss den §.\n"
    article_number = 1
    
    for title, _ in top_articles:
        section_data = law_data.get(title, {})
        name = section_data.get("Name", "Unbekanntes Gesetz")
        aggregated_content = section_data.get("Inhalt", [])

        content = " ".join(aggregated_content)
        prompt += f"\n{article_number}. §: {title} von folgendem Erlass: {name}\n"
        prompt += f"   - **Inhalt:** {content.strip()}\n"
        article_number += 1

    prompt += "\n"
    prompt += "Anfrage auf Deutsch beantworten. Prüfe die Anwendbarkeit der einzelnen § genau. Nenne unbedingt immer den § und das dazugehörige Gesetz oder die zugehörige Verordnung. Wenn ein Artikel keine einschlägigen Aussagen enthält, vergiss ihn.\n"
    return prompt
    
def load_stopwords():
    stopwords = set([
        'aber', 'alle', 'als', 'also', 'am', 'an', 'andere', 'auch',
        'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'das',
        'dass', 'dein', 'deine', 'dem', 'den', 'der', 'des', 'dessen',
        'die', 'dies', 'dieser', 'dieses', 'doch', 'dort', 'du', 'durch',
        'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'er', 'es',
        'euer', 'eure', 'für', 'hatte', 'hatten', 'hattest', 'hattet',
        'hier', 'hinter', 'ich', 'ihr', 'ihre', 'im', 'in', 'ist', 'ja',
        'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jener', 'jenes',
        'jetzt', 'kann', 'kannst', 'können', 'könnt', 'machen', 'mein',
        'meine', 'mit', 'muß', 'mußt', 'musst', 'müssen', 'müsst', 'nach',
        'nachdem', 'nein', 'nicht', 'nun', 'oder', 'seid', 'sein', 'seine',
        'sich', 'sie', 'sind', 'soll', 'sollen', 'sollst', 'sollt', 'sonst',
        'soweit', 'sowie', 'und', 'unser', 'unsere', 'unter', 'vom', 'von',
        'vor', 'wann', 'warum', 'was', 'weiter', 'weitere', 'wenn', 'wer',
        'werde', 'werden', 'werdet', 'weshalb', 'wie', 'wieder', 'wieso',
        'wir', 'wird', 'wirst', 'wo', 'woher', 'wohin', 'zu', 'zum', 'zur',
        'über'
    ])
    
    # Add legal specific stopwords
    legal_stops = {
        'artikel', 'art', 'abs', 'paragraph', 'lit', 'ziffer', 'ziff',
        'bzw', 'vgl', 'etc', 'siehe', 'gemäss'
    }
    return stopwords.union(legal_stops)

GERMAN_STOPS = load_stopwords()


def lemmatize_text(text: str) -> List[str]:
    """
    Lemmatize text using spaCy German model.
    Returns list of lemmatized tokens.
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 1]

def create_bm25_index(law_data: Dict) -> Tuple[BM25Okapi, List[Dict]]:
    """
    Create BM25 index from law data using lemmatization.
    """
    documents = []
    document_metadata = [] 
    
    for article_heading, article_data in law_data.items():
        # Combine heading and content
        content = f"{article_heading} {' '.join(article_data.get('Inhalt', []))}"
        
        # Lemmatize the content
        lemmatized_tokens = lemmatize_text(content)
        
        documents.append(lemmatized_tokens)
        document_metadata.append({
            'heading': article_heading,
            'data': article_data
        })
        
    return BM25Okapi(documents), document_metadata

def search_bm25(keywords: List[str], 
                bm25_index: BM25Okapi, 
                document_metadata: List[Dict],
                top_k: int = 20) -> List[Dict]:
    """
    Search BM25 index using lemmatized keywords.
    """
    # Join keywords and lemmatize
    query_text = " ".join(keywords)
    query_tokens = lemmatize_text(query_text)
    
    # Get document scores
    doc_scores = bm25_index.get_scores(query_tokens)
    
    # Get top k documents
    top_k_idx = np.argsort(doc_scores)[-top_k:][::-1]
    
    results = []
    for idx in top_k_idx:
        if doc_scores[idx] > 0:
            results.append({
                "article": document_metadata[idx],
                "score": doc_scores[idx]
            })
            
    return results
class KeywordExtractionResponse:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords

def extract_keywords_with_llm(user_query):
    prompt = f"""Extrahiere das wichtigste juristische Schlüsselwort aus der folgenden Anfrage.
Fokussiere dich auf das absolut zentrale Thema der Frage und versuche, es zu abstrahieren, aber werde nicht zu generisch, z.B. "Schule".
Gib nur den wichtigsten Begriff zurück, der den Kern der Anfrage trifft.
Gib nur ein Wort oder eine kurze Phrase zurück.
Gib keine Liste zurück.
Anfrage: "{user_query}"
Schlüsselwort:"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein System, das auf die Extraktion juristischer Fachbegriffe aus Anfragen spezialisiert ist.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        
        if completion.choices:
            # Extract the keyword from the response
            keyword = completion.choices[0].message.content.strip()
            return [keyword]
        return []
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


def main_app():
    st.image(logo_path, width=400)
    st.subheader("Abfrage des Thurgauer Schulrechts")
    
    # Initialize session state variables
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""
    if 'top_articles' not in st.session_state:
        st.session_state['top_articles'] = []
    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False
    if 'bm25_index' not in st.session_state:
        print("Creating BM25 index...")
        st.session_state['bm25_index'], st.session_state['document_metadata'] = create_bm25_index(law_data)
        print("BM25 index created successfully")

    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200)

    if user_query:
        print(f"\nProcessing query: {user_query}")
        
        # Direct BM25 search with full query
        print("Performing BM25 search...")
        bm25_results = search_bm25(
            [user_query], # Pass full query
            st.session_state["bm25_index"],
            st.session_state["document_metadata"],
            top_k=20 # Get top 20 results
        )

        # Semantic search
        print("\nPerforming semantic search...")
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        semantic_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Found {len(semantic_articles)} semantic search results")

        # Get titles from semantic results for filtering
        semantic_titles = {title for title, _ in semantic_articles}
        
        st.session_state['top_articles'] = semantic_articles + [(r['article']['heading'], r['score']) 
                                                              for r in bm25_results]

    if st.button("Relevante Bestimmungen"):
        st.session_state.submitted = True
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Semantische Suche") 
            print("\nDisplaying semantic search results...")
            for title, score in semantic_articles:
                title, all_paragraphs, law_name, law_url = get_article_content(title, law_data)
                law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                if law_url:
                    law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"
                st.markdown(f"**{title} - {law_name_display}**", unsafe_allow_html=True)
                if all_paragraphs:
                    for paragraph in all_paragraphs:
                        st.write(paragraph)
                else:
                    st.write("Kein Inhalt verfügbar.")
                    
        with col2:
            st.subheader("Keyword-basierte Suche")
            for result in bm25_results:
                title = result['article']['heading']
                title, all_paragraphs, law_name, law_url = get_article_content(title, law_data)
                law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                if law_url:
                    law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"
                st.markdown(f"**{title} - {law_name_display}**", unsafe_allow_html=True)
                if all_paragraphs:
                    for paragraph in all_paragraphs:
                        st.write(paragraph)
                else:
                    st.write("Kein Inhalt verfügbar.")

    st.write("")
    st.write("")
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Mit GPT 4o beantworten") and user_query:
            if user_query != st.session_state['last_question']:
                prompt = generate_prompt(user_query, st.session_state['top_articles'], law_data)
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                        {"role": "user", "content": prompt}
                    ]
                )

            if response.choices:
                ai_message = response.choices[0].message.content
                st.session_state['last_question'] = user_query
                st.session_state['last_answer'] = ai_message
        else:
            ai_message = st.session_state['last_answer']

        if st.session_state['last_answer']:
            st.subheader("Antwort subsumrary:")
            st.write(st.session_state['last_answer'])
        else:
            st.warning("Bitte geben Sie eine Anfrage ein.")

    with col2:
        if st.button("Prompt generieren und in die Zwischenablage kopieren"):
            if user_query and st.session_state['top_articles']:
                prompt = generate_prompt(user_query, st.session_state['top_articles'], law_data)
                st.session_state['prompt'] = prompt

                html_with_js = generate_html_with_js(prompt)
                html(html_with_js)

                st.text_area("Prompt:", prompt, height=300)
            else:
                if not user_query:
                    st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state['top_articles']:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")
