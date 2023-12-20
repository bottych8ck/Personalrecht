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
    "Gemeindeversammlung": "Die Frage bezieht sich auf Gemeindeversammlungen.",
    "Urnenwahl": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "nicht relevant": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
}


# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)
load_dotenv()  # This line loads the variables from .env

def welcome_page():
    st.title("ChatG-TG für Gemeinderecht")

    # Explanation of what the app does
    st.write("""
        Diese Applikation dient dazu, Anfragen zum Thurgauer Gesetz über das Stimm- und Wahlrecht zu bearbeiten. 
    """)
    st.header("So funktionierts:")
    st.markdown("""
    - Die User stellen eine Anfrage zum Thurgauer Gemeinderecht. 
    - Die Applikation berechnet und zeigt die am besten zur Anfrage passenden Bestimmungen des Gesetzes über das Stimm- und Wahlrecht.
    - Auf der Grundlage der fünf am besten passenden Bestimmungen wird anschliessend ein Prompt für ein sog. Large Language Model (LLM, z.B. ChatGTP) erzeugt. Dieser Prompt beinhaltet wichtige Informationen, die das LLM für die Beantwortung nutzen kann.  
    - Die User können den Prompt in die Zwischenablage kopieren und dem von ihnen genutzten LLM vorlegen.      
    """)
    st.header("Nutzungshinweise")
    st.markdown("""
    - Die Applikation basiert auf der sog. RAG-Technik (Retrieval Augmented Generation). Dabei werden einem LLM bei einer Anfrage passende Informationen vorgelegt, die für die Beantwortung genutzt werden können.
    - Aus Kostengründen erfolgt keine direkte Beantwortung der Frage in der Applikation, weshalb die User den Prompt lediglich kopieren und ihn danach selbst einem LLM vorlegen können.   
    - Der Datenschutz kann gegenwärtig nicht garantiert werden. Verwenden Sie daher keine Personendaten in Ihrer Anfrage.
    - Die Applikation liefert eine Übersicht der semantisch und kontextuell am besten auf die Anfrage passenden Bestimmungen und generiert daraus einen Prompt. Weder die tatsächliche Anwendbarkeit der ausgewählten Bestimmungen noch die Richtigkeiten der Antwort des LLM kann garantiert werden.    
    - Selbst bei Fragen, die nicht direkt das Gemeinderecht des Kantons Thurgau betreffen, sucht das System nach den am besten übereinstimmenden Bestimmungen innerhalb dieses Rechtsbereichs. Beachten Sie jedoch, dass in solchen Fällen die ausgewählten Bestimmungen möglicherweise nicht zutreffend oder relevant sind.
    """)
   

    # Agree button to proceed to the main app
    if st.button("Einverstanden"):
        st.session_state.agreed_to_terms = True

api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)


def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    tags = section_data.get("tags", [])
    if relevance == 'Gemeindeversammlung':
        return any("Assembly" in tag for tag in tags)
    elif relevance == 'Urnenwahl':
        return any("Mail Voting" in tag for tag in tags)
    else:  # If relevance is 'none' or any other value, consider all articles
        return True

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
        article_vector = np.array(article_vector).reshape(1, -1)
        similarity = cosine_similarity(query_vector, article_vector)[0][0]
        similarities[title] = similarity

    return similarities


def get_article_content(title, data):
    # Retrieve the section data for the given title
    section_data = data.get(title, {})

    # Retrieve the list of paragraphs from the section data
    paragraphs = section_data.get('Inhalt', [])

    # Return paragraphs as a list
    return paragraphs

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
        article = law_data.get(title, {})
        name = article.get("Name", "Unbekanntes Gesetz")
        content = ' '.join(article.get("Inhalt", []))

        # Check direct applicability based on user's choice
        if relevance == "Gemeindeversammlung":
            applicability = "Dieser § ist direkt auf Gemeindeversammlungen anwendbar." if "Directly Applicable: Assembly" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Gemeindeversammlungen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        elif relevance == "Urnenwahl":
            applicability = "Dieser § ist direkt auf Urnenwahl anwendbar." if "Directly Applicable: Mail Voting" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Urnenwahlen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        else:
            applicability = ""

        prompt += f"\n{article_number}. §: {title} von folgendem Erass: {name}\n"
        if applicability:
            prompt += f"   - Anwendbarkeit: {applicability}\n"
        prompt += f"   - **Inhalt:** {content}\n"
        article_number += 1

    prompt += "\n\nAnfrage auf Deutsch beantworten\n"

    return prompt



def main_app():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Gemeindeversammlung", "Urnenwahl", "nicht relevant"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Gemeindeversammlungen oder Urnenwahlen bezieht, oder ob dies nicht relevant ist:", relevance_options)

    # Initialize session state variables if they don't exist
    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = []
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # "Abschicken" button to display top matching articles
    if st.button("Abschicken"):
        st.session_state.submitted = True  # Set the flag to True when clicked
        if user_query:
            # Process the query for top articles
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]  # Get only top 5 articles
            st.session_state.top_articles = sorted_articles  # Update session state
            st.write("Die folgenden Artikel werden angezeigt, nachdem Ihre Anfrage analysiert und mit den relevanten Gesetzesdaten abgeglichen wurde. Dieser Prozess funktioniert ähnlich wie eine intelligente Suche, bei der die Bedeutung Ihrer Worte erkannt und die passendsten Inhalte aus den Gesetzestexten ausgewählt werden. Die Bestimmungen müssen aber genau auf ihre tatächliche Anwendbarkeit hin überprüft werden. Diese Überprüfung kann durch ein LLM (Large Language Model) unterstützt werden. Im generierten Prompt sind entsprechende Anweisungen enthalten.")

            with st.expander("Am besten auf die Anfrage passende Artikel", expanded=False):
                for title, score in st.session_state.top_articles:
                    # Retrieve the content of the article using the get_article_content function
                    article_content = get_article_content(title, law_data)  # Correctly passing the title and law_data
                    if article_content:  # Check if there is content available for the article
                        st.write(f" {title}:")  # Display the article title
                        for paragraph in article_content:  # Display each paragraph of the article
                            st.write(paragraph)
                    else:
                        st.write(f"§ {title}: Kein Inhalt verfügbar.")  # Indicate if no content is available for the article
                    st.write("")  # Add a space after each article
        else:
            st.warning("Bitte geben Sie eine Anfrage ein.")
            
    if st.session_state.submitted:
        if st.button("Prompt generieren"):
            if user_query and st.session_state.top_articles:
                # Generate and display the prompt
                prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data)
                st.text_area("Prompt:", prompt, height=300)
                
                # Button to copy the prompt to clipboard
                html_with_js = generate_html_with_js(prompt)
                html(html_with_js)
                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                        {"role": "user", "content": prompt}
                    ]
                )

                if response and response.choices:
                    ai_message = response.choices[0].message.content
                    st.write(f"Antwort: {ai_message}")

            else:
                if not user_query:
                    st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state.top_articles:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")


def main():
    if 'agreed_to_terms' not in st.session_state:
        st.session_state.agreed_to_terms = False

    if not st.session_state.agreed_to_terms:
        welcome_page()
    else:
        main_app()

if __name__ == "__main__":
    main()




