import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    # Normalize the relevance criteria to ensure consistent case comparison
    normalized_relevance = relevance.lower().replace("sek ii", "SEK II")
    
    # Extract and normalize tags from the article data
    tags = section_data.get("tags", [])
    normalized_tags = [tag.lower().replace("sek ii", "SEK II") for tag in tags]
    
    # Define normalized relevance criteria for comparison
    relevance_criteria = ""
    if normalized_relevance == 'staatspersonal':
        relevance_criteria = "staatspersonal"
    elif normalized_relevance == 'lehrperson vs':
        relevance_criteria = "lehrperson vs"
    elif normalized_relevance == 'lehrperson sek ii':
        relevance_criteria = "lehrperson sek ii"

    # Check if any of the normalized tags match the normalized relevance criteria
    return any(relevance_criteria in tag for tag in normalized_tags)


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
            # Ensure article_vector is a numpy array
            article_vector = np.asarray(article_vector, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(query_vector, article_vector)[0][0]
            similarities[title] = similarity
        except TypeError as e:
            print(f"Error processing article '{title}': {e}")
            # Optionally, skip this article or handle the error differently
            continue

    return similarities



def get_article_content(title, law_data):
    # Retrieve the section data for the given title
    section_data = law_data.get(title, {})
    
    law_name = section_data.get("Name", "Unbekanntes Gesetz")  # Default law name
    law_url = section_data.get("URL", "")  # Default to an empty string if no URL is available
    all_paragraphs = section_data.get('Inhalt', [])  # Get the main content paragraphs

    # Check if "Im § erwähnter Artikel des EOG" list exists and append it to the content
    eog_articles = section_data.get("Im § erwähnter Artikel des EOG", [])
    if eog_articles:  # If the list is not empty
        all_paragraphs += ["Im § erwähnter Artikel des EOG:"] + eog_articles  # Append the list as part of the content

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

    
def generate_prompt(user_query, relevance, top_articles, law_data):
    prompt = f"Beantworte folgende Frage: \"{user_query}\"\n\n"
    prompt += "Beantworte die Frage nur gestützt auf einen oder mehrere der folgenden §. Prüfe zuerst, ob der § überhaupt auf die Frage anwendbar ist. Wenn er nicht anwendbar ist, vergiss den §.\n"
    prompt += f"{relevance_mapping.get(relevance, 'Die Frage ist allgemein.')} \n\n"
    article_number = 1

    for title, _ in top_articles:
        section_data = law_data.get(title, {})
        name = "Unbekanntes Gesetz"
        aggregated_content = []
        aggregated_tags = set()

        aggregated_content = section_data.get("Inhalt", [])
        aggregated_tags = set(section_data.get("tags", []))
        name = section_data.get("Name", "Unbekanntes Gesetz")

        content = " ".join(aggregated_content)
        tags = list(aggregated_tags)

        prompt += f"\n{article_number}. §: {title} von folgendem Erlass: {name}\n"
        prompt += f"   - Anwendbarkeit: {applicability}\n"
        prompt += f"   - **Inhalt:** {content.strip()}\n"
        article_number += 1

    prompt += "\nAnswer in German. If a § doesn't say anything relevant to the question don't mention it in your answer. If a directly applicable article says something contrary to an indirectly applicable article, always follow the directly applicable article.\n"
    prompt += "Anfrage auf Deutsch beantworten. Versuche, eine kurze Antwort zu schreiben, prüfe aber die Anwendbarkeit der § genau. Wenn ein Artikel keine einschlägigen Aussagen enthält, erwähne ihn in der Antwort nicht\n"
    return prompt



def main_app():
    st.title("Chat_TG Personalrecht")
    st.subheader("Abfrage des Thurgauer Personalrechts")
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Staatspersonal", "Lehrperson VS", "Lehrperson Sek II"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Staatspersonal, Lehrpersonen der Volksschule oder Lehrpersonen der Berufsfach- und Mittelschulen bezieht:", relevance_options)

    # Initialize session state variables if they don't exist
    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = []
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # "Abschicken" button to display top matching articles
    if st.button("Abschicken"):
        st.session_state.submitted = True  # Set the flag to True when clicked
        if user_query:
            
            query_vector = np.array(get_embeddings(user_query)).reshape(1, -1)            
            similarities = calculate_similarities(query_vector, article_embeddings)
            
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            filtered_articles = [(title, score) for title, score in sorted_articles if is_relevant_article(law_data[title], relevance)]
                        
            st.session_state.top_articles = filtered_articles[:10] 
            st.write("Die folgenden Artikel werden angezeigt, nachdem Ihre Anfrage analysiert und mit den relevanten Gesetzesdaten abgeglichen wurde. Dieser Prozess funktioniert ähnlich wie eine intelligente Suche, bei der die Bedeutung Ihrer Worte erkannt und die passendsten Inhalte aus den Gesetzestexten ausgewählt werden. Die Bestimmungen müssen aber genau auf ihre tatächliche Anwendbarkeit hin überprüft werden. Diese Überprüfung kann durch ein LLM (Large Language Model) unterstützt werden. Im generierten Prompt sind entsprechende Anweisungen enthalten.")
            with st.expander("Am besten auf die Anfrage passende Artikel", expanded=False):
                for title, score in st.session_state.top_articles:
                    # Retrieve the content of the article and the law name using the get_article_content function
                    result = get_article_content(title, law_data)  # Adjusted to handle both standalone and grouped articles
                    _, article_content, law_name, law_url = result
                    law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                    if law_url:
                        law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"
                        
                    st.markdown(f"**{title} - {law_name_display}**", unsafe_allow_html=True)
                     
                    if article_content:
                        for paragraph in article_content:
                            st.write(paragraph)
                    else:
                         st.write("Kein Inhalt verfügbar.")
                    st.write("")
        else:
            st.warning("Bitte geben Sie eine Anfrage ein.")
            
    if st.session_state.submitted:
        if st.button("Prompt generieren"):
            if user_query and st.session_state.top_articles:
                # Generate and display the prompt
                prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data)
                st.text_area("Prompt:", prompt, height=300)
                st.session_state['prompt'] = prompt
                # Button to copy the prompt to clipboard
                html_with_js = generate_html_with_js(prompt)
                html(html_with_js)
            
            else:
                if not user_query:
                    st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state.top_articles:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")
    # if st.button("Antwort anzeigen"):
    #     if st.session_state['prompt']:
     #        response = client.chat.completions.create(
     #            model="gpt-4-1106-preview",
     #            messages=[
     #                {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
    #                 {"role": "user", "content": st.session_state['prompt']}  # Use the prompt from session state
    #             ]
    #         )
     #
     #        if response and response.choices:
     #            ai_message = response.choices[0].message.content
    #             st.write(f"{ai_message}")
   #      else:
   #          st.warning("Bitte generieren Sie zuerst den Prompt.")

def main():
     #if 'agreed_to_terms' not in st.session_state:
       #  st.session_state.agreed_to_terms = False

     #if not st.session_state.agreed_to_terms:
       #  welcome_page()
     #else:
        main_app()

if __name__ == "__main__":
    main()




