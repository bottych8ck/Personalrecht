import streamlit as st
import json
import time
import numpy as np
import os
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# German-specific stopwords and common legal terms to ignore
GERMAN_STOPS = set(stopwords.words('german')).union({
    'artikel', 'art', 'abs', 'paragraph', 'lit', 'ziffer', 'ziff',
    'bzw', 'vgl', 'etc', 'siehe', 'gemäss', 'nach', 'über', 'zum',
    'zur', 'bei', 'sowie', 'oder', 'und', 'der', 'die', 'das'
})

# Configure the page
st.set_page_config(page_title="Fragen zum Bundes-Migrationsrecht", layout="wide")

# Configure Gemini with environment variable
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

def generate_answer(query_text, articles):
    system_prompt = "Sie sind ein juristischer Experte. Beantworte die Frage des Nutzers basierend auf den bereitgestellten Artikeln. Sei präzise und zitieren Sie die relevanten Artikel, wenn möglich."
    
    articles_text = ""
    for article in articles:
        article_text = f"Artikelüberschrift: {article['heading']}\nInhalt: {article['data']['content']}\n\n"
        articles_text += article_text
    
    final_prompt = f"{system_prompt}\n\nFrage des Nutzers: {query_text}\n\nRelevante Artikel:\n{articles_text}"
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(final_prompt)
    return response.text.strip()



def preprocess_german_text(text):
    """Preprocess German text for BM25"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace German umlauts and ß
    text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    
    # Remove punctuation but keep hyphens between words
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Remove standalone numbers but keep numbers that are part of words
    text = re.sub(r'\b\d+\b', '', text)
    
    # Tokenize
    tokens = word_tokenize(text, language='german')
    
    # Remove stopwords and short tokens
    tokens = [token for token in tokens if token not in GERMAN_STOPS and len(token) > 1]
    
    return tokens

def create_bm25_index(law_data):
    """Create BM25 index from law data with German-specific processing"""
    documents = []
    document_metadata = []
    
    for law_name, articles in law_data.items():
        for article_heading, article_data in articles.items():
            # Combine article heading and content for search
            full_text = f"{article_heading} {article_data['content']}"
            
            # Preprocess text
            tokens = preprocess_german_text(full_text)
            
            documents.append(tokens)
            document_metadata.append({
                'heading': article_heading,
                'data': article_data
            })
    
    bm25 = BM25Okapi(documents)
    return bm25, document_metadata

def search_bm25(query, bm25_index, document_metadata, top_k=10):
    """Search using BM25 with German-specific processing"""
    # Preprocess query
    query_tokens = preprocess_german_text(query)
    
    # Get document scores
    doc_scores = bm25_index.get_scores(query_tokens)
    
    # Get top k documents
    top_k_idx = np.argsort(doc_scores)[-top_k:][::-1]
    
    results = []
    for idx in top_k_idx:
        if doc_scores[idx] > 0:  # Only include relevant documents
            results.append({
                'article': document_metadata[idx],
                'score': doc_scores[idx]
            })
    
    return results

# def main():
#     st.title("Fragen zum Bundes-Migrationsrecht")

#     # Initialize session state
#     if 'analyzed_articles' not in st.session_state:
#         st.session_state.analyzed_articles = None
#     if 'query_text' not in st.session_state:
#         st.session_state.query_text = ""

#     try:
#         # Load data
#         law_data, summary_embedding_data = load_data()

#         # Prepare chapter embeddings
#         chapter_embeddings = []
#         for law_full_name, sections in summary_embedding_data.items():
#             for section_title, data in sections.items():
#                 embedding = np.array(data['embedding'])
#                 chapter_embeddings.append({
#                     'law_full_name': law_full_name,
#                     'section_title': section_title,
#                     'embedding': embedding
#                 })

#         # Create articles mapping
#         articles_by_law_and_section = {}
#         for law_full_name, articles in law_data.items():
#             articles_by_section = {}
#             for article_heading, article_data in articles.items():
#                 section = article_data.get('Section', None)
#                 if section is None:
#                     continue
#                 if section not in articles_by_section:
#                     articles_by_section[section] = []
#                 articles_by_section[section].append({
#                     'heading': article_heading,
#                     'data': article_data
#                 })
#             articles_by_law_and_section[law_full_name] = articles_by_section

#         # Query input
#         query_text = st.text_input("Hier Ihre Frage zum Migrationsrecht eingeben:", st.session_state.query_text)

#         if query_text:
#             st.session_state.query_text = query_text
            
#             # Analyze button
#             if st.button("Analysieren"):
#                 with st.spinner("Analyzing..."):
#                     # Get query embedding
#                     query_embedding = get_embedding(query_text)

#                     # Compute similarities
#                     similarities = []
#                     for chapter in chapter_embeddings:
#                         sim = cosine_similarity(query_embedding, chapter['embedding'])
#                         similarities.append({
#                             'law_full_name': chapter['law_full_name'],
#                             'section_title': chapter['section_title'],
#                             'similarity': sim
#                         })

#                     # Sort chapters
#                     similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
#                     top_chapters = similarities[:5]

#                     # Display top chapters
#                     st.subheader("Relevanteste Kapitel und Abschnitte:")
#                     for i, top_chapter in enumerate(top_chapters):
#                         st.write(f"Rank {i+1}: Law: {top_chapter['law_full_name']}, Section: {top_chapter['section_title']}, Similarity: {top_chapter['similarity']:.4f}")

#                     # Get articles
#                     articles_to_evaluate = []
#                     for top_chapter in top_chapters:
#                         law_full_name = top_chapter['law_full_name']
#                         section_title = top_chapter['section_title']
#                         articles_in_section = articles_by_law_and_section.get(law_full_name, {}).get(section_title, [])
#                         articles_to_evaluate.extend(articles_in_section)

#                     # Collect referenced articles
#                     all_articles = collect_articles_with_references(articles_to_evaluate, law_data)
#                     st.session_state.analyzed_articles = all_articles

#                     # Display articles
#                     st.subheader("Relevante Bestimmungen:")
#                     cols = st.columns(2)
                    
#                     with cols[0]:
#                         st.markdown("**Artikel**")
#                     with cols[1]:
#                         st.markdown("**Inhalt**")
                    
#                     for article in all_articles:
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             # Create clickable URL using article ID and URL
#                             article_id = article['data']['ID']
#                             article_url = article['data']['URL']
#                             st.markdown(f"[{article_id}]({article_url})")
                        
#                         with col2:
#                             with st.expander("Show Content"):
#                                 st.write(article['data']['content'])
                    
#                     # st.subheader("Relevant Articles:")
#                     # for article in all_articles:
#                     #     st.write(f"- {article['heading']}")
#                     #     with st.expander("Show Content"):
#                     #         st.write(article['data']['content'])

#             # Generate Answer button
#             if st.session_state.analyzed_articles and st.button("Frage beantworten"):
#                 with st.spinner("Generating answer..."):
#                     answer = generate_answer(query_text, st.session_state.analyzed_articles)
#                     st.subheader("Antwort:")
#                     st.write(answer)

#     except Exception as e:
#         st.error(f"Error: {str(e)}")
def main():
    st.title("Juristischer Assistent")

    # Initialize session state
    if 'analyzed_articles' not in st.session_state:
        st.session_state.analyzed_articles = None
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""
    if 'bm25_index' not in st.session_state:
        st.session_state.bm25_index = None
    if 'document_metadata' not in st.session_state:
        st.session_state.document_metadata = None

    try:
        # Load data
        law_data, summary_embedding_data = load_data()

        # Create BM25 index if not already created
        if st.session_state.bm25_index is None:
            with st.spinner("Erstelle Suchindex..."):
                bm25_index, document_metadata = create_bm25_index(law_data)
                st.session_state.bm25_index = bm25_index
                st.session_state.document_metadata = document_metadata

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

        # Query input
        query_text = st.text_input("Geben Sie Ihre rechtliche Frage ein:", st.session_state.query_text)

        if query_text:
            st.session_state.query_text = query_text
            
            if st.button("Analysieren"):
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

                    # Get semantic search articles
                    articles_to_evaluate = []
                    for top_chapter in top_chapters:
                        law_full_name = top_chapter['law_full_name']
                        section_title = top_chapter['section_title']
                        articles_in_section = articles_by_law_and_section.get(law_full_name, {}).get(section_title, [])
                        articles_to_evaluate.extend(articles_in_section)

                    semantic_articles = collect_articles_with_references(articles_to_evaluate, law_data)
                    
                    # Keyword Search (BM25)
                    bm25_results = search_bm25(query_text, st.session_state.bm25_index, st.session_state.document_metadata)
                    keyword_articles = [result['article'] for result in bm25_results]
                    
                    # Store all articles in session state
                    st.session_state.analyzed_articles = semantic_articles
                    
                    # Create sets of article IDs for comparison
                    semantic_ids = {article['data']['ID'] for article in semantic_articles}
                    keyword_articles_filtered = [article for article in keyword_articles 
                                              if article['data']['ID'] not in semantic_ids]

                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Semantische Suche:")
                        for article in semantic_articles:
                            st.markdown(f"[{article['data']['ID']}]({article['data']['URL']})")
                            with st.expander("Inhalt anzeigen"):
                                st.write(article['data']['content'])
                    
                    with col2:
                        st.subheader("Keyword-basierte Suche:")
                        for article in keyword_articles_filtered:
                            st.markdown(f"[{article['data']['ID']}]({article['data']['URL']})")
                            with st.expander("Inhalt anzeigen"):
                                st.write(article['data']['content'])

            if st.session_state.analyzed_articles and st.button("Antwort generieren"):
                with st.spinner("Generiere Antwort..."):
                    answer = generate_answer(query_text, st.session_state.analyzed_articles)
                    st.subheader("Antwort:")
                    st.write(answer)

    except Exception as e:
        st.error(f"Fehler: {str(e)}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
