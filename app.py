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
from pydantic import BaseModel


# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)

load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'
api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)

class KeywordExtractionResponse(BaseModel):
    keywords: List[str]

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
    """Tokenizer with stemming using NLTK's GermanStemmer."""
    # Initialize the stemmer
    stemmer = GermanStemmer()
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
        # Stem words
        stemmed_words = [stemmer.stem(word) for word in words]
        tokens.extend(stemmed_words)
    return tokens


def create_bm25_index(law_data):
    import time
    start_time = time.time()
    documents = []
    document_metadata = []

    for article_heading, article_data in law_data.items():
        # Combine the 'Inhalt' list into a single string
        content = " ".join(article_data.get("Inhalt", []))
        
        # Create the full text with the article heading and combined content
        full_text = f"{article_heading} {content}"
        
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

def extract_keywords_with_llm(user_query):
    prompt = f"""Extrahiere die wichtigsten rechtlichen Schlüsselbegriffe aus der folgenden Anfrage. Beschränke dich auf das wichtigste Thema der Frage.  
Gib die Schlüsselbegriffe als Liste von Strings im Feld "keywords" zurück. Füge auch relevante Synonyme hinzu.

Anfrage: "{user_query}"
"""

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",  # Use the appropriate model version
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein System, das rechtliche Schlüsselbegriffe aus Anfragen extrahiert.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=KeywordExtractionResponse,
            temperature=0.0,  # Use a low temperature for consistent results
        )

        if completion.choices:
            keywords = completion.choices[0].message.parsed.keywords
            return keywords
        else:
            return []
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


def search_bm25(keywords, bm25_index, document_metadata, top_k=20):
    """Search using BM25 with a list of distilled keywords."""
    # Tokenize keywords (which are already extracted terms)
    stemmer = GermanStemmer()
    query_tokens = [stemmer.stem(word.lower()) for word in keywords]

    # Get document scores
    doc_scores = bm25_index.get_scores(query_tokens)

    # Get top k documents
    top_k_idx = np.argsort(doc_scores)[-top_k:][::-1]

    results = []
    for idx in top_k_idx:
        if doc_scores[idx] > 0:  # Only include relevant documents
            results.append(
                {
                    "article": document_metadata[idx],
                    "score": doc_scores[idx],
                }
            )

    return results

# functions = [
#     {
#         "name": "adjust_keywords",
#         "description": "Suggest new keywords to improve the search results.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "new_keywords": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "A list of new keywords to use in the search.",
#                 },
#             },
#             "required": ["new_keywords"],
#         },
#     },
#     {
#         "name": "stop_search",
#         "description": "Indicates that the search should be concluded.",
#         "parameters": {
#             "type": "object",
#             "properties": {},
#         },
#     },
# ]

functions = [
    {
        "name": "adjust_keywords",
        "description": "Schlage neue Schlüsselbegriffe vor, um die Suchergebnisse zu verbessern.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Eine Liste neuer Schlüsselbegriffe für die Suche.",
                },
            },
            "required": ["new_keywords"],
        },
    },
    {
        "name": "stop_search",
        "description": "Zeigt an, dass die Suche abgeschlossen werden soll.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


def evaluate_bm25_results_with_function_calling(user_query, extracted_keywords, bm25_results, previous_keywords):
    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": """You are an assistant that evaluates search results and suggests new legal keywords if necessary.
Always respond by calling either the 'adjust_keywords' function or the 'stop_search' function. Do not provide any other output.

Your task is to evaluate whether the BM25 search results are relevant to the user's query.

If the results are relevant or sufficient, you should signal the end of the search by calling the 'stop_search' function.

If the results are not relevant or insufficient, you should suggest new legal keywords or synonyms by calling the 'adjust_keywords' function with the new keywords.

Important: Do not suggest keywords that have already been used. Focus on single words or short legal terms.

Do not output anything else besides calling the functions.""",
        },
        {
            "role": "user",
            "content": f"""The user asked:
"{user_query}"

The extracted keywords are: {", ".join(extracted_keywords)}.

The previously used keywords are: {", ".join(previous_keywords)}.

The BM25 search results with these keywords are:""",
        },
    ]

    if bm25_results:
        for idx, result in enumerate(bm25_results[:5]):  # Limit to top 5 for brevity
            article_heading = result['article']['heading']
            article_content = " ".join(result['article']['data'].get("Inhalt", []))[:200]  # First 200 chars
            score = result['score']
            messages.append({
                "role": "user",
                "content": f"{idx+1}. {article_heading} (Score: {score})\nContent: {article_content}"
            })
    else:
        messages.append({
            "role": "user",
            "content": "No results found."
        })

    messages.append({
        "role": "user",
        "content": """Please evaluate whether these results are relevant to the user's question.

If the results are relevant or sufficient, please signal the end of the search by calling the 'stop_search' function.

If the results are not relevant or insufficient, please suggest new legal keywords or synonyms by calling the 'adjust_keywords' function with the new keywords.

Important: Always respond by calling either the 'adjust_keywords' function or the 'stop_search' function. Do not output anything else."""
    })

    # Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.0,
    )

    # Process the response
    message = response.choices[0].message

    # Debugging: Print the assistant's response
    print("Assistant's response:", message)

    if message.function_call:
        function_name = message.function_call.name
        function_arguments = message.function_call.arguments
        try:
            arguments = json.loads(function_arguments)
            new_keywords = arguments.get("new_keywords")
        except Exception as e:
            new_keywords = None
        if function_name == "adjust_keywords" and new_keywords:
            return {"adjust_keywords": True, "new_keywords": new_keywords, "stop": False}
        elif function_name == "stop_search":
            return {"adjust_keywords": False, "new_keywords": None, "stop": True}
        else:
            return {"adjust_keywords": False, "new_keywords": None, "stop": True}
    else:
        return {"adjust_keywords": False, "new_keywords": None, "stop": True}



# def evaluate_bm25_results_with_function_calling(user_query, extracted_keywords, bm25_results):
#     # Prepare the messages
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an assistant that evaluates search results and suggests new keywords if necessary.",
#         },
#         {
#             "role": "user",
#             "content": f"""The user asked:
# "{user_query}"

# The extracted keywords are: {", ".join(extracted_keywords)}.

# The BM25 search results with these keywords are:""",
#         },
#     ]

#     for idx, result in enumerate(bm25_results):
#         article_heading = result['article']['heading']
#         score = result['score']
#         messages.append({
#             "role": "user",
#             "content": f"{idx+1}. {article_heading} (Score: {score})"
#         })

#     messages.append({
#         "role": "user",
#         "content": """Bitte bewerte, ob diese Ergebnisse relevant für die Frage des Nutzers sind. Wenn nicht genügend relevante Ergebnisse vorhanden sind oder keine Ergebnisse gefunden wurden, kannst du eine der folgenden Aktionen ausführen:
    
#     - Wenn die Ergebnisse nicht relevant sind oder keine Ergebnisse vorhanden sind, schlage neue Schlüsselbegriffe vor, indem du die Funktion 'adjust_keywords' mit den neuen Schlüsselbegriffen aufrufst.
#     - Wenn die Ergebnisse relevant sind oder keine weiteren Anpassungen erforderlich sind, signalisiere das Ende der Suche, indem du die Funktion 'stop_search' aufrufst.
    
#     **Wichtig:** Verwende immer eine der Funktionen 'adjust_keywords' oder 'stop_search', um deine Entscheidung mitzuteilen.
#     """
#     })


#     # Call the LLM
#     response = client.chat.completions.create(
#         model="gpt-4o-2024-08-06",
#         messages=messages,
#         functions=functions,
#         function_call="auto",
#     )

#     # Process the response
#     message = response.choices[0].message

#     if message.function_call:
#         function_name = message.function_call.name
#         function_arguments = message.function_call.arguments
#         try:
#             arguments = json.loads(function_arguments)
#             # st.write("Parsed Arguments:", arguments)
#             new_keywords = arguments.get("new_keywords")
#             # st.write("New Keywords:", new_keywords)
#         except Exception as e:
#             # st.write(f"Error parsing function arguments: {e}")
#             new_keywords = None
#         if function_name == "adjust_keywords":
#             arguments = json.loads(function_arguments)
#             new_keywords = arguments.get("new_keywords")
#             return {"adjust_keywords": True, "new_keywords": new_keywords, "stop": False}
#         elif function_name == "stop_search":
#             return {"adjust_keywords": False, "new_keywords": None, "stop": True}
#     else:
#     # No function call, assume adjustment needed
#         return {"adjust_keywords": True, "new_keywords": None, "stop": False}




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
        
        # **Keyword Extraction Step**
        distilled_keywords = extract_keywords_with_llm(user_query)

        if distilled_keywords:
            st.write("**Extrahierte Schlüsselbegriffe:**", ", ".join(distilled_keywords))
        else:
            st.write("Keine Schlüsselbegriffe gefunden.")
            
       # **Iterative BM25 Search with LLM Evaluation**
        previous_keywords = distilled_keywords.copy()
        accumulated_bm25_results = []

        max_iterations = 3
        current_iteration = 1
        
        while current_iteration <= max_iterations:
            st.write(f"**Neue Suche {current_iteration}**")
            
            # **BM25 Search with Distilled Keywords**
            bm25_results = search_bm25(
                distilled_keywords,
                st.session_state["bm25_index"],
                st.session_state["document_metadata"],
            )
            existing_titles = set([result['article']['heading'] for result in accumulated_bm25_results])
            new_bm25_results = [result for result in bm25_results if result['article']['heading'] not in existing_titles]

        #   Accumulate the new results
            accumulated_bm25_results.extend(new_bm25_results)
            
            if not bm25_results:
                st.write("Keine Ergebnisse aus der BM25-Suche.")
            else:
                st.write("**BM25-Suchergebnisse:**")
                for result in bm25_results:
                    st.write(f"- {result['article']['heading']} (Score: {result['score']})")
            
            # **Evaluate BM25 Results and Adjust Keywords if Necessary**
            adjustment_response = evaluate_bm25_results_with_function_calling(
                user_query, distilled_keywords, bm25_results, previous_keywords
            )
            

            if adjustment_response is None:
                st.write("Fehler bei der Verarbeitung der Anpassungsantwort.")
                break
            
            if adjustment_response.get("adjust_keywords"):
                new_keywords = adjustment_response.get("new_keywords")
                if new_keywords:
                    new_keywords = [kw for kw in new_keywords if kw not in previous_keywords]
                    if not new_keywords:
                        st.write("Der Assistent hat keine neuen Schlüsselbegriffe vorgeschlagen.")
                        break  # Exit loop if no new keywords are provided
                    distilled_keywords = new_keywords
                    previous_keywords.extend(new_keywords)
                    st.write("**Neue Schlüsselbegriffe:**", ", ".join(distilled_keywords))

                else:
                    st.write("Der Assistent hat keine neuen Schlüsselbegriffe vorgeschlagen.")
                    break  # Exit loop if no new keywords are provided
                current_iteration += 1
                continue
            elif adjustment_response.get("stop"):
                st.write("Die Suche wurde abgeschlossen.")
                break
            else:
                st.write("Keine weiteren Anpassungen erforderlich.")
                break  # No adjustment needed, proceed
        bm25_results = accumulated_bm25_results

      
        # Semantic search
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        semantic_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
        

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
            for result in bm25_results:  # Display all BM25 results without filtering

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

