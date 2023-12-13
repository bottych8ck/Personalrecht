import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
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
    tags = section_data.get("tags", [])
    if relevance == 'assembly':
        return any("Assembly" in tag for tag in tags)
    elif relevance == 'mail voting':
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
    paragraphs = data.get(title, [])

    # Combine the title and the paragraphs into a single text string
    return title + '\n' + ' '.join(paragraphs)

def generate_prompt(user_query, relevance, top_articles, lawcontent_dict):
    relevance_mapping = {
        "assembly": "Die Frage bezieht sich auf Gemeindeversammlungen.",
        "mail voting": "Die Frage bezieht sich auf Wahlen an der Urne.",
        "none": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
    }

    prompt = f"Beantworte folgende Frage: \"{user_query}\"\n\n"
    prompt += "Beantworte die Frage nur gestützt auf einen oder mehrere der folgenden §. Prüfe zuerst, ob der § überhaupt auf die Frage anwendbar ist. Wenn er nicht anwendbar ist, vergiss den §.\n"
    prompt += f"{relevance_mapping.get(relevance, 'Die Frage ist allgemein.')} \n\n"
    article_number = 1

    for title, _ in top_articles:
        article = lawcontent_dict.get(title, {})
        name = article.get("Name", "Unbekanntes Gesetz")
        content = ' '.join(article.get("Inhalt", []))

        # Check direct applicability based on user's choice
        if relevance == "assembly":
            applicability = "Dieser § ist direkt auf Gemeindeversammlungen anwendbar." if "Directly Applicable: Assembly" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Gemeindeversammlungen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        elif relevance == "mail voting":
            applicability = "Dieser § ist direkt auf Urnenwahl anwendbar." if "Directly Applicable: Mail Voting" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Urnenwahlen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        else:
            applicability = "Überprüfung der direkten Anwendbarkeit ist nicht erforderlich."

        prompt += f"\n{article_number}. §: {title} von folgemden Gesetz:{name}\n   - Anwendbarkeit: {applicability}\n   - **Inhalt:** {content}\n"
        article_number += 1

    prompt += "\n\n**Anfrage basierend auf den obigen Informationen beantworten:**\n"

    return prompt

def main():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["assembly", "mail voting", "none"]
    relevance = st.selectbox("Select relevance:", relevance_options)

    # Generate prompt button
    if st.button("Generate Prompt"):
        if user_query:
            # Process the query
            enhanced_user_query = user_query + " " + relevance_phrases.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(lawcontent_dict, relevance)
            similarities = calculate_similarities(query_vector, {title: embeddings_dict[title] for title in relevant_lawcontent_dict if title in embeddings_dict})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, lawcontent_dict)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

