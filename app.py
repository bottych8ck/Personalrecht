import openai
import os
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

# Commenting out the current tokenize_text function
# def tokenize_text(text):
#     """Improved tokenizer with better handling of German umlauts"""
#     stemmer = GermanStemmer()
#     sentences = re.split(r'[.!?]\s+|\n', text)
#     tokens = []
#     
#     print(f"Tokenizing: {text}")  # Debug print
#     
#     for sentence in sentences:
#         # Convert to lowercase
#         sentence = sentence.lower()
#         
#         # Split on words
#         words = re.findall(r'\b\w+\b', sentence)
#         
#         for word in words:
#             if word not in GERMAN_STOPS and len(word) > 1:
#                 # Create both original and normalized versions
#                 normalized_word = word.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
#                 
#                 # Stem both versions
#                 stemmed_original = stemmer.stem(word)
#                 stemmed_normalized = stemmer.stem(normalized_word)
#                 
#                 # Add both versions to tokens
#                 tokens.append(stemmed_original)
#                 if stemmed_original != stemmed_normalized:
#                     tokens.append(stemmed_normalized)
#                 
#                 print(f"Word: {word} -> Original stem: {stemmed_original}, Normalized stem: {stemmed_normalized}")  # Debug
#     
#     return tokens

# Commenting out the current create_bm25_index function
# def create_bm25_index(law_data):
#     import time
#     start_time = time.time()
#     documents = []
#     document_metadata = []

#     for article_heading, article_data in law_data.items():
#         # Combine the 'Inhalt' list into a single string
#         content = " ".join(article_data.get("Inhalt", []))
#         
#         # Create the full text with the article heading and combined content
#         full_text = f"{article_heading} {content}"
#         
#         # Tokenize text
#         tokens = tokenize_text(full_text)
#         
#         documents.append(tokens)
#         document_metadata.append({
#             'heading': article_heading,
#             'data': article_data
#         })
#     
#     bm25 = BM25Okapi(documents)
#     return bm25, document_metadata

# Commenting out the current search_bm25 function
# def search_bm25(keywords, bm25_index, document_metadata, top_k=20):
#     """Improved BM25 search with better keyword handling"""
#     print(f"Searching with keywords: {keywords}")
#     
#     stemmer = GermanStemmer()
#     query_tokens = []
#     
#     for word in keywords:
#         # Split compound words
#         parts = word.lower().split()
#         for part in parts:
#             # Create both original and normalized versions
#             normalized_part = part.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
#             
#             # Stem both versions
#             stemmed_original = stemmer.stem(part)
#             stemmed_normalized = stemmer.stem(normalized_part)
#             
#             # Add both versions to query tokens
#             query_tokens.append(stemmed_original)
#             if stemmed_original != stemmed_normalized:
#                 query_tokens.append(stemmed_normalized)
#             
#             print(f"Original: {part} -> Stemmed original: {stemmed_original}, Stemmed normalized: {stemmed_normalized}")

#     # Get document scores
#     doc_scores = bm25_index.get_scores(query_tokens)
#     
#     print(f"Number of documents with non-zero scores: {sum(1 for score in doc_scores if score > 0)}")
#     
#     # Get top k documents
#     top_k_idx = np.argsort(doc_scores)[-top_k:][::-1]
#     
#     results = []
#     for idx in top_k_idx:
#         if doc_scores[idx] > 0:  # Only include relevant documents
#             results.append({
#                 "article": document_metadata[idx],
#                 "score": doc_scores[idx],
#             })
#             print(f"Found document: {document_metadata[idx]['heading']} with score {doc_scores[idx]}")
#     
#     return results

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

# 2. Update extract_keywords_with_llm function to not use beta.chat.completions.parse
def extract_keywords_with_llm(user_query):
    prompt = f"""Extract the main legal keywords from the following query.  
    Focus on the absolutely primary topic of the question and try to abstract. Return only the most important term that goes to the core of the query.
    Return between 1-5 keywords as a comma-separated list.
    Query: "{user_query}"
    Keywords:"""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a system specialized in extracting legal terminology from queries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        
        if completion.choices:
            # Extract keywords from the response and split into list
            keywords = [k.strip() for k in completion.choices[0].message.content.split(',')]
            return keywords
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
        
        # **Keyword Extraction Step**
        print("\nExtracting keywords...")
        distilled_keywords = extract_keywords_with_llm(user_query)

        if distilled_keywords:
            print(f"Extracted keywords: {', '.join(distilled_keywords)}")
            st.write("**Extrahierte Schlüsselbegriffe:**", ", ".join(distilled_keywords))
        else:
            print("No keywords found")
            st.write("Keine Schlüsselbegriffe gefunden.")
            
        # **Iterative BM25 Search with LLM Evaluation**
        previous_keywords = distilled_keywords.copy()
        accumulated_bm25_results = []

        max_iterations = 3
        current_iteration = 1
        
        while current_iteration <= max_iterations:
            print(f"\nIteration {current_iteration}")
            st.write(f"**Suche {current_iteration}**")
            
            # **BM25 Search with Distilled Keywords**
            print(f"Performing BM25 search with keywords: {distilled_keywords}")
            bm25_results = search_bm25(
                distilled_keywords,
                st.session_state["bm25_index"],
                st.session_state["document_metadata"],
            )
            
            existing_titles = set([result['article']['heading'] for result in accumulated_bm25_results])
            new_bm25_results = [result for result in bm25_results if result['article']['heading'] not in existing_titles]
            print(f"Found {len(new_bm25_results)} new results")

            # Accumulate the new results
            accumulated_bm25_results.extend(new_bm25_results)
            
            if not bm25_results:
                print("No BM25 results found")
                st.write("Keine Ergebnisse aus der BM25-Suche.")
            else:
                print("BM25 search results:")
                st.write("**BM25-Suchergebnisse:**")
                for result in bm25_results:
                    print(f"- {result['article']['heading']} (Score: {result['score']})")
                    st.write(f"- {result['article']['heading']} (Score: {result['score']})")
            
            # **Evaluate BM25 Results and Adjust Keywords if Necessary**
            print("\nEvaluating BM25 results...")
            adjustment_response = evaluate_bm25_results_with_function_calling(
                user_query, distilled_keywords, bm25_results, previous_keywords
            )
            
            if adjustment_response is None:
                print("Error processing adjustment response")
                st.write("Fehler bei der Verarbeitung der Anpassungsantwort.")
                break
            
            if adjustment_response.get("adjust_keywords"):
                new_keywords = adjustment_response.get("new_keywords")
                if new_keywords:
                    new_keywords = [kw for kw in new_keywords if kw not in previous_keywords]
                    if not new_keywords:
                        print("No new keywords suggested")
                        st.write("Der Assistent hat keine neuen Schlüsselbegriffe vorgeschlagen.")
                        break
                    distilled_keywords = new_keywords
                    previous_keywords.extend(new_keywords)
                    print(f"New keywords: {', '.join(new_keywords)}")
                    st.write("**Neue Schlüsselbegriffe:**", ", ".join(distilled_keywords))
                else:
                    print("No new keywords provided")
                    st.write("Der Assistent hat keine neuen Schlüsselbegriffe vorgeschlagen.")
                    break
                current_iteration += 1
                continue
            elif adjustment_response.get("stop"):
                print("Search completed")
                st.write("Die Suche wurde abgeschlossen.")
                break
            else:
                print("No further adjustments needed")
                st.write("Keine weiteren Anpassungen erforderlich.")
                break

        print(f"\nTotal accumulated results: {len(accumulated_bm25_results)}")
        bm25_results = accumulated_bm25_results
        
        print("\nFiltering relevant articles...")
        bm25_relevant_articles = filter_relevant_articles(user_query, bm25_results)
        print(f"Found {len(bm25_relevant_articles)} relevant articles after filtering")

        # Semantic search
        print("\nPerforming semantic search...")
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        semantic_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Found {len(semantic_articles)} semantic search results")

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
            for result in bm25_relevant_articles:  # Display all BM25 results without filtering

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
if __name__ == "__main__":
    main_app()







# def tokenize_text(text):
#     """Improved tokenizer with better handling of German umlauts"""
#     stemmer = GermanStemmer()
#     sentences = re.split(r'[.!?]\s+|\n', text)
#     tokens = []
    
#     print(f"Tokenizing: {text}")  # Debug print
    
#     for sentence in sentences:
#         # Convert to lowercase
#         sentence = sentence.lower()
        
#         # Split on words
#         words = re.findall(r'\b\w+\b', sentence)
        
#         for word in words:
#             if word not in GERMAN_STOPS and len(word) > 1:
#                 # Create both original and normalized versions
#                 normalized_word = word.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
                
#                 # Stem both versions
#                 stemmed_original = stemmer.stem(word)
#                 stemmed_normalized = stemmer.stem(normalized_word)
                
#                 # Add both versions to tokens
#                 tokens.append(stemmed_original)
#                 if stemmed_original != stemmed_normalized:
#                     tokens.append(stemmed_normalized)
                
#                 print(f"Word: {word} -> Original stem: {stemmed_original}, Normalized stem: {stemmed_normalized}")  # Debug
    
#     return tokens

# def create_bm25_index(law_data):
#     import time
#     start_time = time.time()
#     documents = []
#     document_metadata = []

#     for article_heading, article_data in law_data.items():
#         # Combine the 'Inhalt' list into a single string
#         content = " ".join(article_data.get("Inhalt", []))
        
#         # Create the full text with the article heading and combined content
#         full_text = f"{article_heading} {content}"
        
#         # Tokenize text
#         tokens = tokenize_text(full_text)
        
#         documents.append(tokens)
#         document_metadata.append({
#             'heading': article_heading,
#             'data': article_data
#         })
    
#     bm25 = BM25Okapi(documents)
#     return bm25, document_metadata




# def search_bm25(keywords, bm25_index, document_metadata, top_k=20):
#     """Improved BM25 search with better keyword handling"""
#     print(f"Searching with keywords: {keywords}")
    
#     stemmer = GermanStemmer()
#     query_tokens = []
    
#     for word in keywords:
#         # Split compound words
#         parts = word.lower().split()
#         for part in parts:
#             # Create both original and normalized versions
#             normalized_part = part.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
            
#             # Stem both versions
#             stemmed_original = stemmer.stem(part)
#             stemmed_normalized = stemmer.stem(normalized_part)
            
#             # Add both versions to query tokens
#             query_tokens.append(stemmed_original)
#             if stemmed_original != stemmed_normalized:
#                 query_tokens.append(stemmed_normalized)
            
#             print(f"Original: {part} -> Stemmed original: {stemmed_original}, Stemmed normalized: {stemmed_normalized}")

#     # Get document scores
#     doc_scores = bm25_index.get_scores(query_tokens)
    
#     print(f"Number of documents with non-zero scores: {sum(1 for score in doc_scores if score > 0)}")
    
#     # Get top k documents
#     top_k_idx = np.argsort(doc_scores)[-top_k:][::-1]
    
#     results = []
#     for idx in top_k_idx:
#         if doc_scores[idx] > 0:  # Only include relevant documents
#             results.append({
#                 "article": document_metadata[idx],
#                 "score": doc_scores[idx],
#             })
#             print(f"Found document: {document_metadata[idx]['heading']} with score {doc_scores[idx]}")
    
#     return results

# functions = [
#     {
#         "name": "adjust_keywords",
#         "description": "Schlage neue Schlüsselbegriffe vor, um die Suchergebnisse zu verbessern.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "new_keywords": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "Eine Liste neuer Schlüsselbegriffe für die Suche.",
#                 },
#             },
#             "required": ["new_keywords"],
#         },
#     },
#     {
#         "name": "stop_search",
#         "description": "Zeigt an, dass die Suche abgeschlossen werden soll.",
#         "parameters": {
#             "type": "object",
#             "properties": {},
#         },
#     },
# ]

# def evaluate_bm25_results_with_function_calling(user_query, extracted_keywords, bm25_results, previous_keywords):
#     # Prepare the messages
#     messages = [
#         {
#             "role": "system",
#             "content": """You are an assistant that evaluates search results and suggests new legal keywords if necessary.
# Always respond by calling either the 'adjust_keywords' function or the 'stop_search' function. Do not provide any other output.

# Your task is to evaluate whether the BM25 search results are relevant to the user's query.

# If the results are relevant or sufficient, you should signal the end of the search by calling the 'stop_search' function.

# If the results are not relevant or insufficient, you should suggest new legal keywords or synonyms by calling the 'adjust_keywords' function with the new keywords.

# Important: Do not suggest keywords that have already been used. Focus on single words or short legal terms.

# Do not output anything else besides calling the functions.""",
#         },
#         {
#             "role": "user",
#             "content": f"""The user asked:
# "{user_query}"

# The extracted keywords are: {", ".join(extracted_keywords)}.

# The previously used keywords are: {", ".join(previous_keywords)}.

# The BM25 search results with these keywords are:""",
#         },
#     ]

#     if bm25_results:
#         for idx, result in enumerate(bm25_results[:5]):  # Limit to top 5 for brevity
#             article_heading = result['article']['heading']
#             article_content = " ".join(result['article']['data'].get("Inhalt", []))
#             score = result['score']
#             messages.append({
#                 "role": "user",
#                 "content": f"{idx+1}. {article_heading} (Score: {score})\nContent: {article_content}"
#             })
#     else:
#         messages.append({
#             "role": "user",
#             "content": "No results found."
#         })

#     messages.append({
#         "role": "user",
#         "content": """Please evaluate whether these results are relevant to the user's question.

# If the results are relevant or sufficient, please signal the end of the search by calling the 'stop_search' function.

# If the results are not relevant or insufficient, please suggest new legal keywords or synonyms by calling the 'adjust_keywords' function with the new keywords.

# Important: Always respond by calling either the 'adjust_keywords' function or the 'stop_search' function. Do not output anything else."""
#     })

#     # Call the LLM
#     response = openai_client.chat.completions.create(
#         model="gpt-4o-2024-08-06",
#         messages=messages,
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#     )

#     # Process the response
#     message = response.choices[0].message

#     # Debugging: Print the assistant's response
#     print("Assistant's response:", message)

#     if message.function_call:
#         function_name = message.function_call.name
#         function_arguments = message.function_call.arguments
#         try:
#             arguments = json.loads(function_arguments)
#             new_keywords = arguments.get("new_keywords")
#         except Exception as e:
#             new_keywords = None
#         if function_name == "adjust_keywords" and new_keywords:
#             return {"adjust_keywords": True, "new_keywords": new_keywords, "stop": False}
#         elif function_name == "stop_search":
#             return {"adjust_keywords": False, "new_keywords": None, "stop": True}
#         else:
#             return {"adjust_keywords": False, "new_keywords": None, "stop": True}
#     else:
#         return {"adjust_keywords": False, "new_keywords": None, "stop": True}
# def filter_relevant_articles(user_query, articles):
#     """
#     Filter relevant articles and add debug logging
#     """
#     print(f"Starting to filter {len(articles)} articles")
    
#     # Define the function schema for article relevance evaluation
#     evaluation_schema = {
#         "name": "evaluate_articles",
#         "description": "Evaluate the relevance of legal articles to a user query",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "evaluations": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "heading": {
#                                 "type": "string",
#                                 "description": "The heading of the article being evaluated"
#                             },
#                             "is_relevant": {
#                                 "type": "boolean",
#                                 "description": "Whether the article is directly relevant to the query"
#                             },
#                             "reason": {
#                                 "type": "string",
#                                 "description": "Brief explanation of why the article is relevant or not"
#                             }
#                         },
#                         "required": ["heading", "is_relevant"]
#                     }
#                 }
#             },
#             "required": ["evaluations"]
#         }
#     }

#     relevant_articles = []
#     batch_size = 5
    
#     for i in range(0, len(articles), batch_size):
#         batch = articles[i:i+batch_size]
#         print(f"Processing batch {i//batch_size + 1} with {len(batch)} articles")
        
#         system_message = """You are a legal expert assistant that evaluates the relevance of legal articles to user queries.
# For each article, determine if it contains information that directly helps answer the user's question.
# Only mark an article as relevant if it contains specific, applicable information.
# IMPORTANT: Be more lenient in marking articles as relevant - if an article might contain useful information, mark it as relevant."""

#         user_message = f"""Query: "{user_query}"

# Please evaluate the following articles:

# {format_articles_for_evaluation(batch)}"""

#         try:
#             response = groq_client.chat.completions.create(
#                 model="llama2-70b-4096",
#                 messages=[
#                     {"role": "system", "content": system_message},
#                     {"role": "user", "content": user_message}
#                 ],
#                 functions=[evaluation_schema],
#                 function_call={"name": "evaluate_articles"},
#                 temperature=0.1
#             )
            
#             # Extract and validate the function call response
#             function_call = response.choices[0].message.function_call
#             if function_call and function_call.name == "evaluate_articles":
#                 try:
#                     evaluation_results = json.loads(function_call.arguments)
#                     print(f"Evaluation results: {json.dumps(evaluation_results, indent=2)}")
                    
#                     # Process the validated results
#                     batch_relevant_articles = process_evaluation_results(evaluation_results, batch)
#                     print(f"Found {len(batch_relevant_articles)} relevant articles in this batch")
#                     relevant_articles.extend(batch_relevant_articles)
#                 except json.JSONDecodeError as e:
#                     print(f"Failed to parse function response: {e}")
#                     print(f"Raw response: {function_call.arguments}")
#                     continue
                
#         except Exception as e:
#             print(f"Batch processing failed: {e}")
#             continue

#     print(f"Total relevant articles found: {len(relevant_articles)}")
#     return relevant_articles

# def process_evaluation_results(evaluation_results, batch):
#     """Process and validate the evaluation results with debug logging"""
#     relevant_articles = []
    
#     print("Processing evaluation results:")
#     print(f"Raw evaluation results: {json.dumps(evaluation_results, indent=2)}")
    
#     for evaluation in evaluation_results.get("evaluations", []):
#         if not isinstance(evaluation, dict):
#             print(f"Skipping invalid evaluation format: {evaluation}")
#             continue
            
#         heading = evaluation.get("heading")
#         is_relevant = evaluation.get("is_relevant")
#         reason = evaluation.get("reason", "No reason provided")
        
#         print(f"Evaluating article: {heading}")
#         print(f"Is relevant: {is_relevant}")
#         print(f"Reason: {reason}")
        
#         if heading and isinstance(is_relevant, bool) and is_relevant:
#             # Find the matching article in the batch
#             for article in batch:
#                 if article['article']['heading'] == heading:
#                     print(f"Adding relevant article: {heading}")
#                     relevant_articles.append(article)
#                     break
    
#     print(f"Found {len(relevant_articles)} relevant articles")
#     return relevant_articles

# def format_articles_for_evaluation(batch):
#     """Format the batch of articles for the prompt"""
#     formatted_articles = []
#     for article in batch:
#         heading = article['article']['heading']
#         content = " ".join(article['article']['data'].get("Inhalt", []))[:500]
#         formatted_articles.append(f"Article: {heading}\nContent: {content}")
#     return "\n\n".join(formatted_articles)

# def process_evaluation_results(evaluation_results, batch):
#     """Process and validate the evaluation results"""
#     relevant_articles = []
    
#     for evaluation in evaluation_results.get("evaluations", []):
#         if not isinstance(evaluation, dict):
#             continue
            
#         heading = evaluation.get("heading")
#         is_relevant = evaluation.get("is_relevant")
        
#         if heading and isinstance(is_relevant, bool) and is_relevant:
#             # Find the matching article in the batch
#             for article in batch:
#                 if article['article']['heading'] == heading:
#                     relevant_articles.append(article)
#                     break
    
#     return relevant_articles


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
