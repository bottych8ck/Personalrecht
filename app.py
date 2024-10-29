import streamlit as st
import json
import time
import numpy as np
import os
import google.generativeai as genai

# Configure the page
st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

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

def main():
    st.title("Legal RAG Assistant")

    # Initialize session state
    if 'analyzed_articles' not in st.session_state:
        st.session_state.analyzed_articles = None
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""

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

        # Query input
        query_text = st.text_input("Enter your legal question:", st.session_state.query_text)

        if query_text:
            st.session_state.query_text = query_text
            
            # Analyze button
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    # Get query embedding
                    query_embedding = get_embedding(query_text)

                    # Compute similarities
                    similarities = []
                    for chapter in chapter_embeddings:
                        sim = cosine_similarity(query_embedding, chapter['embedding'])
                        similarities.append({
                            'law_full_name': chapter['law_full_name'],
                            'section_title': chapter['section_title'],
                            'similarity': sim
                        })

                    # Sort chapters
                    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
                    top_chapters = similarities[:5]

                    # Display top chapters
                    st.subheader("Most Relevant Sections:")
                    for i, top_chapter in enumerate(top_chapters):
                        st.write(f"Rank {i+1}: Law: {top_chapter['law_full_name']}, Section: {top_chapter['section_title']}, Similarity: {top_chapter['similarity']:.4f}")

                    # Get articles
                    articles_to_evaluate = []
                    for top_chapter in top_chapters:
                        law_full_name = top_chapter['law_full_name']
                        section_title = top_chapter['section_title']
                        articles_in_section = articles_by_law_and_section.get(law_full_name, {}).get(section_title, [])
                        articles_to_evaluate.extend(articles_in_section)

                    # Collect referenced articles
                    all_articles = collect_articles_with_references(articles_to_evaluate, law_data)
                    st.session_state.analyzed_articles = all_articles

                    # Display articles
                    st.subheader("Relevant Articles:")
                    for article in all_articles:
                        st.write(f"- {article['heading']}")
                        with st.expander("Show Content"):
                            st.write(article['data']['content'])

            # Generate Answer button
            if st.session_state.analyzed_articles and st.button("Generate Answer"):
                with st.spinner("Generating answer..."):
                    answer = generate_answer(query_text, st.session_state.analyzed_articles)
                    st.subheader("Answer:")
                    st.write(answer)

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
