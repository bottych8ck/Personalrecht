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

    # Check if "Im § erwähnter Artikel des EOG" exists and append its content to all_paragraphs
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
    // Automatically copy to clipboard when the script is loaded
    copyToClipboard();
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
    return prompt

def main_app():
    st.title("Chat_TG Schulrecht")
    st.subheader("Abfrage des Thurgauer Schulrechts")
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""

    user_query = st.text_input("Hier Ihre Frage eingeben:")

    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = []
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
            
        top_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                        
        st.session_state.top_articles = top_articles[:10]
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    with st.expander("am Besten passende Bestimmungen", expanded=True): 
        for title, score in st.session_state.top_articles:
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
                query_vector = get_embeddings(user_query)
                similarities = calculate_similarities(query_vector, article_embeddings)
                top_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)      
                st.session_state.top_articles = top_articles[:10]
                  
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
        else:
            ai_message = st.session_state['last_answer']

    if st.session_state['last_answer']:
        st.subheader("Antwort subsumrary:")
        st.write(st.session_state['last_answer'])
    else:
        st.warning("Bitte geben Sie eine Anfrage ein.")

    with col2:    
        if st.button("Prompt generieren und in die Zwischenablage kopieren"):
            if user_query and st.session_state.top_articles:
                # Generate the prompt
                prompt = generate_prompt(user_query, st.session_state.top_articles, law_data)
                st.session_state['prompt'] = prompt
    
                # Create HTML with JavaScript to copy the prompt to the clipboard
                html_with_js = generate_html_with_js(prompt)
                html(html_with_js)
    
                # Display the generated prompt in a text area
                st.text_area("Prompt:", prompt, height=300)
            else:
                # if not user_query:
                #     st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state.top_articles:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")
          
    if st.button("Hinweise"):
        st.session_state.submitted = True
        st.write("Die folgenden Artikel bilden die Grundlage der obigen Antwort. Sie wurden aufgrund einer Analyse der Anfrage und einem Vergleich und mit den relevanten Gesetzesdaten berechnet.")
        with st.expander("Am besten auf die Anfrage passende Artikel", expanded=False):
            for title, score in st.session_state.top_articles:
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




def main():
    main_app()

if __name__ == "__main__":
    main()
