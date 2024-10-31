import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)

load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'
api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)


def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    normalized_relevance = relevance.lower().replace("sek ii", "SEK II")
    tags = section_data.get("tags", [])
    normalized_tags = [tag.lower().replace("sek ii", "SEK II") for tag in tags]
    
    relevance_criteria = normalized_relevance
    
    is_relevant = any(relevance_criteria in tag for tag in normalized_tags)
    
    return is_relevant

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

def tokenize_text(text):
    """Custom tokenizer using regular expressions to avoid NLTK dependencies."""
    # Split text into sentences based on punctuation marks and newline characters
    sentences = re.split(r'[.!?]\s+|\n', text)
    tokens = []
    for sentence in sentences:
        # Convert to lowercase
        sentence = sentence.lower()
        # Replace German umlauts and ß
        sentence = sentence.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        # Split on words
        words = re.findall(r'\b\w+\b', sentence)
        # Remove stopwords and short tokens
        words = [word for word in words if word not in GERMAN_STOPS and len(word) > 1]
        tokens.extend(words)
    return tokens


import time

def create_bm25_index(law_data):
    start_time = time.time()
    documents = []
    document_metadata = []
    
    for law_name, articles in law_data.items():
        for article_heading, article_data in articles.items():
            # Combine article heading and content for search
            full_text = f"{article_heading} {article_data['content']}"
            
            # Tokenize text
            tokens = tokenize_text(full_text)
            
            documents.append(tokens)
            document_metadata.append({
                'heading': article_heading,
                'data': article_data
            })
    
    bm25 = BM25Okapi(documents)
    st.write(f"BM25 index created in {time.time() - start_time:.2f} seconds")
    return bm25, document_metadata


def search_bm25(query, bm25_index, document_metadata, top_k=20):
    """Search using BM25 with German-specific processing"""
    # Tokenize query
    query_tokens = tokenize_text(query)
    
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
        st.session_state['bm25_index'], st.session_state['document_metadata'] = create_bm25_index(law_data)

    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200)

    if user_query:
        # Semantic search
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        semantic_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # BM25 search
        bm25_results = search_bm25(user_query, st.session_state['bm25_index'], 
                                 st.session_state['document_metadata'])
        
        # Get titles from semantic results for filtering
        semantic_titles = {title for title, _ in semantic_articles}
        
        # Filter BM25 results to remove duplicates
        filtered_bm25_results = [
            result for result in bm25_results 
            if result['article']['heading'] not in semantic_titles
        ][:10]

        st.session_state['top_articles'] = semantic_articles + [(r['article']['heading'], r['score']) 
                                                              for r in filtered_bm25_results]

    if st.button("Relevante Bestimmungen"):
        st.session_state.submitted = True
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Semantische Suche")
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
            for result in filtered_bm25_results:
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
                response = client.chat.completions.create(
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
if __name__ == "__main__":
    main_app()


# def main_app():
#     st.image(logo_path, width=400)
#     st.subheader("Abfrage des Thurgauer Schulrechts")
    
#     # Initialize session state variables
#     if 'last_question' not in st.session_state:
#         st.session_state['last_question'] = ""
#     if 'last_answer' not in st.session_state:
#         st.session_state['last_answer'] = None
#     if 'prompt' not in st.session_state:
#         st.session_state['prompt'] = ""
#     if 'top_articles' not in st.session_state:
#         st.session_state['top_articles'] = []
#     if 'submitted' not in st.session_state:
#         st.session_state['submitted'] = False

#     user_query = st.text_area("Hier Ihre Frage eingeben:", height=200)

#     if user_query:
#         query_vector = get_embeddings(user_query)
#         similarities = calculate_similarities(query_vector, article_embeddings)
#         top_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
#         st.session_state['top_articles'] = top_articles[:10]
#     if st.button("Relevante Bestimmungen"):
#         st.session_state.submitted = True
#         with st.expander("Am besten passende Bestimmungen", expanded=True):
#             for title, score in st.session_state['top_articles']:
#                 title, all_paragraphs, law_name, law_url = get_article_content(title, law_data)
#                 law_name_display = law_name if law_name else "Unbekanntes Gesetz"
#                 if law_url:
#                     law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"
                    
#                 st.markdown(f"**{title} - {law_name_display}**", unsafe_allow_html=True)
#                 if all_paragraphs:
#                     for paragraph in all_paragraphs:
#                         st.write(paragraph)
#                 else:
#                     st.write("Kein Inhalt verfügbar.")

#     st.write("")
#     st.write("")
#     st.write("")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("Mit GPT 4o beantworten") and user_query:
#             if user_query != st.session_state['last_question']:
#                 query_vector = get_embeddings(user_query)
#                 similarities = calculate_similarities(query_vector, article_embeddings)
#                 top_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
#                 st.session_state['top_articles'] = top_articles[:10]
#                 prompt = generate_prompt(user_query, st.session_state['top_articles'], law_data)
#                 response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
#                         {"role": "user", "content": prompt}
#                     ]
#                 )

#             if response.choices:
#                 ai_message = response.choices[0].message.content
#                 st.session_state['last_question'] = user_query
#                 st.session_state['last_answer'] = ai_message
#         else:
#             ai_message = st.session_state['last_answer']

#         if st.session_state['last_answer']:
#             st.subheader("Antwort subsumrary:")
#             st.write(st.session_state['last_answer'])
#         else:
#             st.warning("Bitte geben Sie eine Anfrage ein.")

#     with col2:
#         if st.button("Prompt generieren und in die Zwischenablage kopieren"):
#             if user_query and st.session_state['top_articles']:
#                 prompt = generate_prompt(user_query, st.session_state['top_articles'], law_data)
#                 st.session_state['prompt'] = prompt

#                 html_with_js = generate_html_with_js(prompt)
#                 html(html_with_js)

#                 st.text_area("Prompt:", prompt, height=300)
#             else:
#                 if not user_query:
#                     st.warning("Bitte geben Sie eine Anfrage ein.")
#                 if not st.session_state['top_articles']:
#                     st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")



if __name__ == "__main__":
    main()
