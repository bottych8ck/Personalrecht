import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import requests
from google.cloud import storage
from google.oauth2 import service_account  # Ensure this import is included
import json
from groq import Groq
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()

def get_secret(name):
# Access the secret version
    project_id = "law-data-for-ask-the-tellist"
    secret_name = name
    resource_name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": resource_name})
    return response.payload.data.decode('UTF-8', errors='replace')  # Replace undecodable characters

def get_environment_or_secret(name):
    # First, try to get the secret from environment variables
    secret = os.getenv(name)
    if not secret:
        # If the environment variable is not set, get it from the Secret Manager
        secret = get_secret(name)
    return secret

# Get API keys and credentials
openai_api_key = get_environment_or_secret('OPENAI_API_KEY')
groq_api_key = get_environment_or_secret('GROQ_API_KEY')
google_credentials_json = get_environment_or_secret('GOOGLE_APPLICATION_CREDENTIALS_JSON')

openai_client = openai.OpenAI(api_key=openai_api_key)
groq_client = Groq(api_key=groq_api_key)

google_credentials = json.loads(google_credentials_json)
credentials = service_account.Credentials.from_service_account_info(google_credentials)
client = storage.Client(credentials=credentials)


credentials = service_account.Credentials.from_service_account_info(google_credentials)
client = storage.Client(credentials=credentials)


## Define your bucket and file paths
bucket_name = "data_embeddings_ask"
file1_path = "article_embeddings.json"
file2_path = "knowledge_base_embeddings.json"  # Adjust this based on your file structure

def load_json_from_gcs_as_numpy(bucket_name, file_path):
    try:
        # Load JSON from Google Cloud Storage
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        file_content = blob.download_as_text()  # Download the content as text
        data_dict = json.loads(file_content)  # Parse the JSON content into a dictionary
        
        # Convert dictionary values (embeddings) to NumPy arrays
        numpy_data_dict = {key: np.array(value, dtype=np.float32) for key, value in data_dict.items()}
        
        return numpy_data_dict
    
    except Exception as e:
        st.error(f"Failed to load or parse {file_path} from GCS: {e}")
        st.stop()


article_embeddings = load_json_from_gcs_as_numpy(bucket_name, file1_path)
knowledge_base_embeddings = load_json_from_gcs_as_numpy(bucket_name, file2_path)
    
# Mapping for relevance criteria
relevance_mapping = {
    "Staatspersonal": "Die Frage bezieht sich auf Staatspersonal.",
    "Lehrperson VS": "Die Frage bezieht sich auf Lehrpersonen an der Volksschule.",
    "Lehrperson BfS": "Die Frage bezieht sich auf Lehrpersonen an der Sekundarstufe II."
}
tags_mapping = {
    "directly applicable: Staatspersonal": "Staatspersonal",
    "indirectly applicable: Staatspersonal": "Staatspersonal",
    "directly applicable: Lehrperson VS": "Schulrecht / Lehrperson VS",
    "indirectly applicable: Lehrperson VS": "Schulrecht / Lehrperson VS",
    "directly applicable: Lehrperson Sek II": "Lehrperson Sek II",
    "indirectly applicable: Lehrperson Sek II": "Lehrperson Sek II"
}

reverse_tags_mapping = {
    "Staatspersonal": ["directly applicable: Staatspersonal", "indirectly applicable: Staatspersonal"],
    "Schulrecht / Lehrperson VS": ["directly applicable: Lehrperson VS", "indirectly applicable: Lehrperson VS"],
    "Lehrperson Sek II": ["directly applicable: Lehrperson Sek II", "indirectly applicable: Lehrperson Sek II"]
}


    
with open('law_data.json', 'r', encoding='utf-8', errors='replace') as file:
    law_data = json.load(file)


with open('knowledge_base.json', 'r', encoding='utf-8', errors='replace') as file:
    knowledge_base = json.load(file)


for item_id, item in knowledge_base.items():
    content = item.get("Content", "")
    if isinstance(content, str):
        # Split content into a list of lines
        knowledge_base[item_id]["Content"] = content.split('\n')

load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'


def get_embeddings(text):
    res = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    # Normalize relevance input
    normalized_relevance = relevance.lower().replace("schulrecht / ", "").replace("sek ii", "SEK II")
    
    # Normalize the tags from the section data
    tags = section_data.get("Tags", section_data.get("tags", []))
    normalized_tags = [tag.lower().replace("sek ii", "SEK II") for tag in tags]

    # Check if any of the normalized tags match the normalized relevance criteria
    is_relevant = any(normalized_relevance in tag for tag in normalized_tags)
    
    return is_relevant

def get_relevant_articles(law_data, relevance):
    relevant_articles = {}
    for section, section_data in law_data.items():
        if is_relevant_article(section_data, relevance):
            relevant_articles[section] = section_data
    return relevant_articles


def calculate_similarities(query_vector, article_embeddings):
# Convert the query vector to a NumPy array and ensure it's a 2D array
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    # Stack all article vectors into a single NumPy array for batch processing
    all_article_vectors = np.stack(list(article_embeddings.values()))

    # Compute cosine similarity in a vectorized manner
    # Normalize the vectors
    query_norm = np.linalg.norm(query_vector)
    article_norms = np.linalg.norm(all_article_vectors, axis=1)

    # Dot product between query and all article vectors
    dot_products = np.dot(all_article_vectors, query_vector.T).flatten()

    # Compute cosine similarities
    similarities = dot_products / (query_norm * article_norms)

    # Create a dictionary to store the results
    similarity_dict = dict(zip(article_embeddings.keys(), similarities))

    print("Calculated similarities for", len(similarity_dict), "articles.")    
    return similarity_dict




def get_article_content(uid, law_data):
    # Fetch the article information using the UID
    article_info = law_data.get(uid, {})

    # Extract necessary information from the article
    title = article_info.get('Title', 'Unknown Title')
    all_paragraphs = article_info.get('Inhalt', [])
    law_name = article_info.get("Name", "Unbekanntes Gesetz")
    law_url = article_info.get("URL", "")

    # Check if "Im § erwähnter Artikel des EOG" exists and append its content to all_paragraphs
    mentioned_articles = article_info.get("Im § erwähnter Artikel des EOG", [])
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



def generate_prompt(user_query, relevance, top_articles, law_data, top_knowledge_items):
    prompt = f"Beantworte folgende Frage: \"{user_query}\"\n\n"
    prompt += "Beantworte die Frage nur gestützt auf einen oder mehrere der folgenden §. Prüfe zuerst, ob der § überhaupt auf die Frage anwendbar ist. Wenn er nicht anwendbar ist, vergiss den §.\n"
    prompt += f"{relevance_mapping.get(relevance, '')} \n\n"
    article_number = 1
    
    for uid, _ in top_articles:
        title, all_paragraphs, law_name, law_url = get_article_content(uid, law_data)
        content = " ".join(all_paragraphs)
        prompt += f"\n{article_number}. §: {title} von folgendem Erlass: {law_name}\n"
        prompt += f"   - **Inhalt:** {content.strip()}\n"
        article_number += 1

    prompt += "\n"
    
    prompt += "\nZusätzlich berücksichtige folgende allgemeine Hinweise aus der Beratungspraxis:\n"
    for item_id, _ in top_knowledge_items:
        item = knowledge_base.get(item_id, {})
        title = item.get("Title", "Unbekannt")
        content = ' '.join(item.get("Content", []))
        prompt += f"- {title}: {content}\n"


    prompt += "Anfrage auf Deutsch beantworten. Prüfe die  Anwendbarkeit der einzelnen § genau. Wenn ein Artikel keine einschlägigen Aussagen enthält, vergiss ihn.\n"
    prompt += "Mache nach der Antwort ein Fazit. \n"

    return prompt



def main_app():
    st.image(logo_path, width=400)
    st.subheader("Abfrage des Thurgauer Schul- und Personalrecht und der Telefonliste des Rechtsdiensts")
    # if 'last_question' not in st.session_state:
    #     st.session_state['last_question'] = ""
    # if 'last_answer' not in st.session_state:
    #     st.session_state['last_answer'] = None
    # if 'last_answer_gpt4o' not in st.session_state:
    #     st.session_state['last_answer_gpt4o'] = None
    # if 'top_articles' not in st.session_state:
    #     st.session_state['top_articles'] = []
    # if 'top_knowledge_items' not in st.session_state:
    #     st.session_state['top_knowledge_items'] = []
    # if 'prompt' not in st.session_state:
    #     st.session_state['prompt'] = ""
    # if 'submitted' not in st.session_state:
    #     st.session_state['submitted'] = False
    # if 'show_form' not in st.session_state:
    #     st.session_state['show_form'] = False
    # if 'delete_form' not in st.session_state:
    #     st.session_state['delete_form'] = False
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'top_articles' not in st.session_state:
        st.session_state['top_articles'] = []
    if 'top_knowledge_items' not in st.session_state:
        st.session_state['top_knowledge_items'] = []
    if 'relevance' not in st.session_state:
        st.session_state['relevance'] = "Schulrecht / Lehrperson VS"  # Default relevance
    if 'show_model_selection' not in st.session_state:
        st.session_state['show_model_selection'] = False  # Control flag for model selection visibility



    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200, key="user_query_text_area")

    relevance_options = ["Schulrecht / Lehrperson VS", "Staatspersonal", "Lehrperson Sek II"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Schulrecht oder Lehrpersonen der Volksschule, auf Staatspersonal oder Lehrpersonen der Berufsfach- und Mittelschulen bezieht:", relevance_options)

    if st.button("Bearbeiten"):
        st.session_state['relevance'] = relevance
        st.session_state['last_question'] = user_query
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)

        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        filtered_articles = [(title, score) for title, score in sorted_articles if is_relevant_article(law_data[title], relevance)]
        st.session_state.top_articles = filtered_articles[:10]

        knowledge_similarities = calculate_similarities(query_vector, knowledge_base_embeddings)
        st.session_state.top_knowledge_items = [
            (item_id, score) for item_id, score in sorted(knowledge_similarities.items(), key=lambda x: x[1], reverse=True)
            if is_relevant_article(knowledge_base[item_id], relevance)
        ][:30]


        st.session_state.submitted = True
        with st.expander("Am besten auf die Anfrage passende Bestimmungen und Einträge in der Telefonliste", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Bestimmungen (Top-10)")
                for uid, score in st.session_state.top_articles:
                    article_info = law_data.get(str(uid), None)
                    if article_info:
                        title, all_paragraphs, law_name, law_url = get_article_content(str(uid), law_data)
                        law_name_display = law_name if law_name else "Unbekanntes Gesetz"
                        if law_url:
                            law_name_display = f"<a href='{law_url}' target='_blank'>{law_name_display}</a>"

                        title_clean = title.strip('*')
                        st.markdown(f"**{title_clean} - {law_name_display}**", unsafe_allow_html=True)

                        if all_paragraphs:
                            for paragraph in all_paragraphs:
                                st.write(paragraph)
                        else:
                            st.write("Kein Inhalt verfügbar.")

            with col2:
                st.markdown("#### Einträge in der Telefonliste (Top-30)")
                for item_id, _ in st.session_state.top_knowledge_items:
                    item = knowledge_base.get(item_id, {})
                    title = item.get("Title", "Unbekannt")
                    content = item.get("Content", "")
                    year = item.get("Year", "")
                    
                    if isinstance(content, list):
                        # Join list into a single string with double spaces at the end of each line to ensure Markdown respects line breaks
                        content = '  \n'.join(content)
                    
                    # Display the title and content with proper formatting
                    st.markdown(f"**{title}**")
                    st.markdown(f"*Auskunft aus dem Jahr {year}*")  # Display the year under the title
                    st.markdown(content)




        st.write("")
        st.write("")
        st.write("")   
        st.write("")    


        col1, col2 = st.columns(2)
        with col1:
        
            # if 'last_answer' not in st.session_state:
            #     st.session_state['last_answer'] = ""
            # if 'last_model' not in st.session_state:
            #     st.session_state['last_model'] = ""
            # if 'show_model_selection' not in st.session_state:
            #     st.session_state['show_model_selection'] = False
            # if 'selected_model' not in st.session_state:
            #     st.session_state['selected_model'] = None
            # # Button to start the process
            # if st.button("Antwort mit Sprachmodell"):
                # st.session_state['show_model_selection'] = True
        
            if 'show_model_selection' not in st.session_state:
                st.session_state['show_model_selection'] = False
            if 'selected_model' not in st.session_state:
                st.session_state['selected_model'] = None
            if 'last_answer' not in st.session_state:
                st.session_state['last_answer'] = None
        
            # Button to show model selection
            if st.button("Antwort mit Sprachmodell generieren"):
            #     st.session_state['show_model_selection'] = True
            #     st.experimental_rerun()  # Force a rerun to update the UI
        
            # # Model selection and answer generation
            # if st.session_state['show_model_selection']:
            #     model_selection = st.selectbox(
            #         "Wählen Sie ein Sprachmodell aus:",
            #         ["Llama 3.1", "GPT 4o"]
            #     )
            #     st.session_state['selected_model'] = model_selection
        
            #     if st.button("Antwort generieren"):
                prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)         
        
                        # Handle Llama 3.1 model selection
                if st.session_state['selected_model'] == "Llama 3.1":
                    try:
                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama-3.1-70b-versatile"
                        )
    
                        if chat_completion.choices and len(chat_completion.choices) > 0:
                            ai_message = chat_completion.choices[0].message.content
                            st.session_state['last_answer'] = ai_message
                            st.session_state['last_model'] = "Llama 3.1"
                        else:
                            st.warning("No response generated from Llama 3.1.")
    
                    except groq.InternalServerError as e:
                        st.error(f"An internal server error occurred with the Groq API: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred with the Groq API: {str(e)}")
    
                    # elif st.session_state['selected_model'] == "GPT 4o":
                    #     try:
                    #         response = openai_client.chat.completions.create(
                    #             model="gpt-4o-2024-08-06",
                    #             messages=[
                    #                 {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                    #                 {"role": "user", "content": prompt}
                    #             ]
                    #         )
        
                    #         if response.choices:
                    #             ai_message = response.choices[0].message.content
                    #             st.session_state['last_answer'] = ai_message
                    #             st.session_state['last_model'] = "GPT 4o"
                    #         else:
                    #             st.warning("No response generated from GPT-4o.")
                    #     except Exception as e:
                    #         st.error(f"An error occurred with the OpenAI API: {str(e)}")
        
        
                    # Display the generated answer
                    if st.session_state['last_answer']:
                        st.subheader(f"Antwort subsumary ({st.session_state['last_model']}):")
                        st.write(st.session_state['last_answer'])
                else:
                    st.warning("Please enter a query before generating an answer.")
                    
        with col2:
                
            if st.button("Prompt generieren und in die Zwischenablage kopieren"):
                if user_query and st.session_state.top_articles:
                    # Generate the prompt
                    prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)
                    st.session_state['prompt'] = prompt
        
                    # Create HTML with JavaScript to copy the prompt to the clipboard
                    html_with_js = generate_html_with_js(prompt)
                    st.components.v1.html(html_with_js)
        
                    # Display the generated prompt in a text area
                    st.text_area("Prompt:", prompt, height=300)
                else:
                    if not st.session_state.top_articles:
                        st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")
        

if __name__ == "__main__":
    main_app()



