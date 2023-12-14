import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
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
    # Retrieve the list of paragraphs for the given title, default to an empty list if the title is not found
    paragraphs = law_data.get('inhalt', [])

    # Return paragraphs as a list
    return paragraphs


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

    prompt += "\n\nAnfrage auf Deutsch beantworten:\n"

    return prompt



def main():
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
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            st.session_state.top_articles = sorted_articles[:5]  # Update session state

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in st.session_state.top_articles:
                article_content = get_article_content(title, law_data[title])  # Correctly accessing the 'inhalt' key inside get_article_content
                st.write(f"{title} (Score: {round(score, 2)}):")
                for paragraph in article_content:
                    st.write(paragraph)
                st.write("")  # Add a space after each article
        else:
            st.warning("Bitte geben Sie eine Anfrage ein.")

    if st.session_state.submitted:
        if st.button("Generate Prompt"):
            if user_query and st.session_state.top_articles:
                # Generate and display the prompt
                prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data)
                st.text_area("Generated Prompt:", prompt, height=300)
            else:
                if not user_query:
                    st.warning("Bitte geben Sie eine Anfrage ein.")
                if not st.session_state.top_articles:
                    st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")

if __name__ == "__main__":
    main()




