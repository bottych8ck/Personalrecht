import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Mapping for relevance criteria
relevance_mapping = {
    "Staatspersonal": "Die Frage bezieht sich auf Staatspersonal.",
    "Lehrperson VS": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "Lehrperson BfS": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
}

# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)

load_dotenv()  # This line loads the variables from .env

api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)

def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    normalized_relevance = relevance.lower().replace("sek ii", "SEK II")
    tags = section_data.get("tags", [])
    normalized_tags = [tag.lower().replace("sek ii", "SEK II") for tag in tags]
    
    relevance_criteria = normalized_relevance  # Direct use of normalized_relevance ensures we're checking against the correct criteria
    
    # Check if any of the normalized tags match the normalized relevance criteria
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
    return [(title, all_paragraphs, law_name, law_url)]

def generate_html_with_js(prompt):
    return f"""
    <textarea id='text_area' style='opacity: 0; position: absolute; left: -9999px;'>{prompt}</textarea>
    <button onclick='copyToClipboard()'>Text in die Zwischenablage kopieren</button>
    <script>
    function copyToClipboard() {{
        var copyText = document.getElementById('text_area');
        copyText.style.opacity = 1; // Make the textarea visible to enable selection
        copyText.select();
        document.execCommand('copy');
        copyText.style.opacity = 0; // Hide the textarea again
        alert('Copied to clipboard!');
    }}
    </script>
    """

def generate_prompt(user_query, relevance, top_articles, law_data):
    prompt = f"Beantworte folgende Frage: \"{user_query}\"\n\n"
    prompt += "Beantworte die Frage nur gestützt auf einen oder mehrere der folgenden §. Prüfe zuerst, ob der § überhaupt auf die Frage anwendbar ist. Wenn er nicht anwendbar ist, vergiss den §.\n"
    prompt += f"{relevance_mapping.get(relevance, 'Die Frage ist allgemein.')} \n\n"
    article_number = 1
    
    for title, _ in top_articles:
        section_data = law_data.get(title, {})
        name = section_data.get("Name", "Unbekanntes Gesetz")
        aggregated_content = section_data.get("Inhalt", [])
        name = section_data.get("Name", "Unbekanntes Gesetz")

        content = " ".join(aggregated_content)
        prompt += f"\n{article_number}. §: {title} von folgendem Erlass: {name}\n"
        prompt += f"   - **Inhalt:** {content.strip()}\n"
        article_number += 1

    prompt += "\nAnswer in German. If a § doesn't say anything relevant to the question don't mention it in your answer. If a directly applicable article says something contrary to an indirectly applicable article, always follow the directly applicable article.\n"
    prompt += "Anfrage auf Deutsch beantworten. Versuche, eine kurze Antwort zu schreiben, prüfe aber die Anwendbarkeit der § genau. Wenn ein Artikel keine einschlägigen Aussagen enthält, erwähne ihn in der Antwort nicht.\n"
    return prompt

def main_app():
    st.title("Chat_TG Personalrecht")
    st.subheader("Abfrage des Thurgauer Personalrechts")
    
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""

    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Staatspersonal", "Lehrperson VS", "Lehrperson Sek II"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Staatspersonal, Lehrpersonen der Volksschule oder Lehrpersonen der Berufsfach- und Mittelschulen bezieht:", relevance_options)

    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = []
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    if st.button("Abschicken"):
        st.session_state.submitted = True
        if user_query:
            query_vector = get_embeddings(user_query)
            similarities = calculate_similarities(query_vector, article_embeddings)
            
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            filtered_articles = [(title, score) for title, score in sorted_articles if is_relevant_article(law_data[title], relevance)]
            
            st.session_state.top_articles = filtered_articles[:10]
            st.write("Die folgenden Artikel werden angezeigt, nachdem Ihre Anfrage analysiert und mit den relevanten Gesetzesdaten abgeglichen wurde.")
            with st.expander("Am besten auf die Anfrage passende Artikel", expanded=False):
                for title, score in st.session_state.top_articles:
                    result = get_article_content(title, law_data)
                    article_content, law_name, law_url = result
                    law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                    if law_url:
                        law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"
                        
                    st.markdown(f"**{title} - {law_name_display}**", unsafe_allow_html=True)
                    if article_content:
                        for paragraph in article_content:
                            st.write(paragraph)
                    else:
                        st.write("Kein Inhalt verfügbar.")
        else:
            st.warning("Bitte geben Sie eine Anfrage ein.")
            
    if st.session_state.submitted:
        if st.button("Prompt generieren"):
            if user_query and st.session_state.top_articles:
                prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data)
                st.text_area("Prompt:", prompt, height=300)
                st.session_state['prompt'] = prompt
                html_with_js = generate_html_with_js(prompt)
                components.html(html_with_js, height=100)
            else:
                if not user_query:
                    st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state.top_articles:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")

def main():
    main_app()

if __name__ == "__main__":
    main()
