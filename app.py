import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime


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
    print("Calculated similarities for", len(similarities), "articles.")    
    return similarities

def get_article_content(uid, law_data):
    # Fetch the article information using the UID
    article_info = law_data.get(uid, {})

    # Extract necessary information from the article
    title = article_info.get('Title', 'Unknown Title')
    all_paragraphs = article_info.get('Inhalt', [])
    law_name = article_info.get("Name", "Unbekanntes Gesetz")
    law_url = article_info.get("URL", "")

    return (title, all_paragraphs, law_name, law_url)

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

def generate_prompt(user_query, top_articles, law_data):
    prompt = f"Beantworte folgende Frage: \"{user_query}\"\n\n"
    prompt += "Beantworte die Frage nur gestützt auf einen oder mehrere der folgenden §. Prüfe zuerst, ob der § überhaupt auf die Frage anwendbar ist. Wenn er nicht anwendbar ist, vergiss den §.\n"
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

    prompt += "\n"
    
   
    prompt += "Anfrage auf Deutsch beantworten. Prüfe die  Anwendbarkeit der einzelnen § genau. Wenn ein Artikel keine einschlägigen Aussagen enthält, vergiss ihn.\n"
    prompt += "Mache nach der Antwort ein Fazit und erwähne dort die relevanten § mitsamt dem Erlassnahmen \n"

    return prompt




def main_app():
    st.title("Chat_TG / subsumary")
    st.subheader("(Testversion für die KVTG)")
    st.subheader("Abfrage des Thurgauer Rechtsbuches")
    
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""

    user_query = st.text_input("Hier Ihre Frage eingeben:")

    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = []
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if user_query != st.session_state['last_question']:
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        st.session_state.top_articles = sorted_articles[:10]

    if st.button("Hinweise"):
        st.session_state.submitted = True
        st.write("Die folgenden Bestimmungen  am Besten auf die Anfrage. Nicht alle angezeigten Bestimmungen sind wirklich einschlägig.")
        with st.expander("Bestimmungen und Hinweise", expanded=False):
            st.markdown("#### Bestimmungen")
            for uid, score in st.session_state.top_articles:  # Assuming top_articles stores (uid, score)
                title, all_paragraphs, law_name, law_url = get_article_content(uid, law_data)
                law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                if law_url:
                    law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"
                    
                st.markdown(f"**{title} - {law_name_display}**", unsafe_allow_html=True)
                if all_paragraphs:
                    for paragraph in all_paragraphs:
                        st.write(paragraph)
                else:
                    st.write("Kein Inhalt verfügbar.")
                    
    if st.button("Mit GPT 4o beantworten)") and user_query:
        
        if user_query != st.session_state['last_question']:
            query_vector = get_embeddings(user_query)
            prompt = generate_prompt(user_query, st.session_state.top_articles, law_data)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                    {"role": "user", "content": prompt}
                ]
            )
    
            # Display the response from OpenAI
            if response.choices:
                ai_message = response.choices[0].message.content  # Corrected attribute access
                st.session_state['last_question'] = user_query
                st.session_state['last_answer'] = ai_message
                update_gist_with_query_and_response(user_query, ai_message)
        else:
            ai_message = st.session_state['last_answer']

    if st.session_state['last_answer']:
        st.subheader("Antwort Chat-TG:")
        st.write(st.session_state['last_answer'])
    else:
        st.warning("Bitte geben Sie eine Anfrage ein.")
        
    if st.session_state.submitted:
        if st.button("Prompt generieren"):
            if user_query and st.session_state.top_articles:
                prompt = generate_prompt(user_query, st.session_state.top_articles, law_data)
                html_with_js = generate_html_with_js(prompt)
                html(html_with_js)
                st.text_area("Prompt:", prompt, height=300)
                st.session_state['prompt'] = prompt
                         
            else:
                if not user_query:
                    st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state.top_articles:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")


def main():
    main_app()

if __name__ == "__main__":
    main()

