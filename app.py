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
from groq import Groq


# Mapping for relevance criteria
relevance_mapping = {
    "Staatspersonal": "Die Frage bezieht sich auf Staatspersonal.",
    "Lehrperson VS": "Die Frage bezieht sich auf Lehrpersonen an der Volksschule.",
    "Lehrperson SEK II": "auf Lehrpersonen auf der Sekundarstufe II."
}
tags_mapping = {
    "directly applicable: Staatspersonal": "Staatspersonal",
    "indirectly applicable: Staatspersonal": "Staatspersonal",
    "directly applicable: Lehrperson VS": "Lehrperson VS",
    "indirectly applicable: Lehrperson VS": "Lehrperson VS",
    "directly applicable: Lehrperson Sek II": "Lehrperson Sek II",
    "indirectly applicable: Lehrperson Sek II": "Lehrperson Sek II"
}

reverse_tags_mapping = {
    "Staatspersonal": ["directly applicable: Staatspersonal", "indirectly applicable: Staatspersonal"],
    "Lehrperson VS": ["directly applicable: Lehrperson VS", "indirectly applicable: Lehrperson VS"],
    "Lehrperson Sek II": ["directly applicable: Lehrperson Sek II", "indirectly applicable: Lehrperson Sek II"]
}

# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)
    
with open('law_data.json', 'r') as file:
    law_data = json.load(file)

with open('knowledge_base_embeddings.json', 'r') as file:
    knowledge_base_embeddings = json.load(file)

with open('knowledge_base.json', 'r') as file:
    knowledge_base = json.load(file)
load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'

openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=openai_api_key)

groq_api_key = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=groq_api_key)

def keyword_search(keyword, law_data, knowledge_base):
    keyword = keyword.lower()
    
    matching_articles = {}
    for uid, article in law_data.items():
        if keyword in article.get('Title', '').lower():
            matching_articles[uid] = article
            continue
            
        if any(keyword in paragraph.lower() for paragraph in article.get('Inhalt', [])):
            matching_articles[uid] = article
            
    matching_items = {}
    for item_id, item in knowledge_base.items():
        if keyword in item.get('Title', '').lower():
            matching_items[item_id] = item
            continue
            
        if any(keyword in content.lower() for content in item.get('Content', [])):
            matching_items[item_id] = item
            
    return matching_articles, matching_items
            


def update_file_in_github(file_path, content, commit_message="Update file"):
    repo_owner = os.getenv('GITHUB_REPO_OWNER')
    repo_name = os.getenv('GITHUB_REPO_NAME')
    token = os.getenv('GITHUB_TOKEN')
    branch_name = os.getenv('GITHUB_BRANCH', 'learningsubsumary')  # Assuming 'learningsubsumary' is the branch name

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}?ref={branch_name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get the current file SHA
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    sha = response.json()["sha"]

    # Prepare the data to update the file
    data = {
        "message": commit_message,
        "content": base64.b64encode(content.encode('utf-8')).decode('utf-8'),
        "sha": sha,
        "branch": branch_name
    }

    # Update the file
    response = requests.put(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def update_knowledge_base_local(new_data):
    update_file_in_github('knowledge_base.json', json.dumps(new_data, indent=4, ensure_ascii=False))
    st.success("Knowledge base updated in GitHub.")

def update_knowledge_base_embeddings_local(new_embeddings):
    update_file_in_github('knowledge_base_embeddings.json', json.dumps(new_embeddings, indent=4, ensure_ascii=False))
    st.success("Knowledge base embeddings updated in GitHub.")

def add_to_knowledge_base(title, content, category, tags):
    if knowledge_base:
        max_id = max(int(k) for k in knowledge_base.keys())
    else:
        max_id = 0
    new_id = str(max_id + 1)
    knowledge_base[new_id] = {
        "Title": title,
        "Content": [content],
        "Category": category,
        "Tags": tags  # Ensure tags are stored as a flat list
    }
    update_knowledge_base_local(knowledge_base)
    
    # Create and store the embedding
    embedding = get_embeddings(content)
    knowledge_base_embeddings[new_id] = embedding
    update_knowledge_base_embeddings_local(knowledge_base_embeddings)

def delete_from_knowledge_base(entry_id):
    if entry_id in knowledge_base:
        del knowledge_base[entry_id]
        if entry_id in knowledge_base_embeddings:
            del knowledge_base_embeddings[entry_id]
        update_knowledge_base_local(knowledge_base)
        update_knowledge_base_embeddings_local(knowledge_base_embeddings)
        st.success(f"Entry {entry_id} successfully deleted.")
    else:
        st.error(f"Entry {entry_id} not found.")

def get_embeddings(text):
    res = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    normalized_relevance = relevance.lower().replace("sek ii", "SEK II")
    
    # Try to get "Tags" first (for knowledge_base), fallback to "tags" (for law_data) if not found
    tags = section_data.get("Tags", section_data.get("tags", []))
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
    prompt += f"{relevance_mapping.get(relevance, 'Die Frage ist allgemein.')} \n\n"
    article_number = 1
    
    for uid, _ in top_articles:
        title, all_paragraphs, law_name, law_url = get_article_content(uid, law_data)
        content = " ".join(all_paragraphs)
        prompt += f"\n{article_number}. §: {title} von folgendem Erlass: {law_name}\n"
        prompt += f"   - **Inhalt:** {content.strip()}\n"
        article_number += 1

    prompt += "\n"
    
    prompt += "\nZusätzlich berücksichtige folgende allgemeine Grundsätze und Prinzipien:\n"
    for item_id, _ in top_knowledge_items:
        item = knowledge_base.get(item_id, {})
        title = item.get("Title", "Unbekannt")
        content = ' '.join(item.get("Content", []))
        prompt += f"- {title}: {content}\n"
    prompt += "\n"
    prompt += "\n"
    prompt += "\n"
    prompt += "Anfrage auf Deutsch beantworten. Prüfe die  Anwendbarkeit der einzelnen § genau. Wenn ein Artikel keine einschlägigen Aussagen enthält, vergiss ihn.\n"
    prompt += "Mache nach der Antwort ein Fazit und erwähne dort die relevanten § mitsamt dem Erlassnahmen \n"

    return prompt



def main_app():
    st.image(logo_path, width=400)
    st.subheader("Abfrage des Thurgauer Personalrechts")
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'last_answer_gpt4o' not in st.session_state:
        st.session_state['last_answer_gpt4o'] = None
    if 'top_articles' not in st.session_state:
        st.session_state['top_articles'] = []
    if 'top_knowledge_items' not in st.session_state:
        st.session_state['top_knowledge_items'] = []
    if 'prompt' not in st.session_state:
        st.session_state['generated_prompt'] = ""
    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False
    if 'show_form' not in st.session_state:
        st.session_state['show_form'] = False
    if 'delete_form' not in st.session_state:
        st.session_state['delete_form'] = False
    if 'generating_answer' not in st.session_state:
        st.session_state.generating_answer = False
    if 'start_generating_answer' not in st.session_state:
        st.session_state.start_generating_answer = False

    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200, key="user_query_text_area")

    relevance_options = ["Staatspersonal", "Lehrperson VS", "Lehrperson Sek II"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Staatspersonal, Lehrpersonen der Volksschule oder Lehrpersonen der Berufsfach- und Mittelschulen bezieht:", relevance_options)

    
    if st.button("Bearbeiten"):
        st.session_state['relevance'] = relevance
        st.session_state['last_question'] = user_query
        st.session_state.generating_answer = False
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        filtered_articles = [(title, score) for title, score in sorted_articles if is_relevant_article(law_data[title], relevance)]
        st.session_state.top_articles = filtered_articles[:10]
        knowledge_similarities = calculate_similarities(query_vector, knowledge_base_embeddings)
        st.session_state.top_knowledge_items = [
            (item_id, score) for item_id, score in sorted(knowledge_similarities.items(), key=lambda x: x[1], reverse=True)
            if is_relevant_article(knowledge_base[item_id], relevance)
        ][:1]

        st.session_state['last_question'] = user_query
        st.session_state.submitted = True
    if st.session_state.get('submitted'):
        with st.expander("Am besten auf die Anfrage passende Bestimmungen und Wissenselemente", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Bestimmungen")
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
                st.markdown("#### Wissenselemente")
                for item_id, _ in st.session_state.top_knowledge_items:
                    item = knowledge_base.get(item_id, {})
                    title = item.get("Title", "Unbekannt")
                    content = ' '.join(item.get("Content", []))
                    st.markdown(f"**{title}**")
                    st.write(content)
                            
    if st.session_state.get('submitted'):
        st.write("### Hier können Sie eine Stichwortsuche durchführen und auswählen, welche Resultate für die Beantwortung berücksichtigt werden:")
        keyword = st.text_input("Stichwort eingeben und Enter drücken:")
        
        if keyword:
            matching_articles, matching_items = keyword_search(keyword, law_data, knowledge_base)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Gefundene Gesetzesartikel")
                selected_article_uids = []
                
                for uid, article in matching_articles.items():
                    title = article.get('Title', 'Unknown Title')
                    law_name = article.get('Name', 'Unbekanntes Gesetz')
                    content = article.get('Inhalt', [])
                    
                    # Create container for each article
                    with st.container():
                        # Checkbox with title and law name
                        col_check, col_expand = st.columns([8,48])
                        with col_check:
                            if st.checkbox(f"{title}", key=f"article_{uid}"):
                                selected_article_uids.append(uid)
                            st.markdown(f"<small style='color: gray;'>{law_name}</small>", unsafe_allow_html=True)
                        
                        # Expander for content
                        with col_expand:
                            with st.expander("📖"):
                                for paragraph in content:
                                    st.write(paragraph)
                        st.markdown("---")
                
                if selected_article_uids and st.button("Ausgewählte Artikel hinzufügen"):
                    existing_uids = [uid for uid, _ in st.session_state.top_articles]
                    for uid in selected_article_uids:
                        if uid not in existing_uids:
                            st.session_state.top_articles.append((uid, 1.0))
                    st.success("Ausgewählte Artikel wurden zu den relevanten Artikeln hinzugefügt")
            
            with col2:
                st.markdown("#### Gefundene Wissenselemente")
                selected_item_ids = []
                
                for item_id, item in matching_items.items():
                    title = item.get('Title', 'Unknown Title')
                    content = ' '.join(item.get('Content', []))
                    
                    # Create container for each knowledge item
                    with st.container():
                        col_check, col_expand = st.columns([8,1])
                        with col_check:
                            if st.checkbox(f"{title}", key=f"item_{item_id}"):
                                selected_item_ids.append(item_id)
                        
                        # Expander for content
                        with col_expand:
                            with st.expander("📖"):
                                st.write(content)
                        st.markdown("---")
                
                if selected_item_ids and st.button("Ausgewählte Wissenselemente hinzufügen"):
                    existing_ids = [item_id for item_id, _ in st.session_state.top_knowledge_items]
                    for item_id in selected_item_ids:
                        if item_id not in existing_ids:
                            st.session_state.top_knowledge_items.append((item_id, 1.0))
                    st.success("Ausgewählte Wissenselemente wurden zu den relevanten Wissenselementen hinzugefügt")           
        if 'show_form' not in st.session_state:
            st.session_state.show_form = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Neues Wissenselement hinzufügen"):
                st.session_state.show_form = not st.session_state.show_form
                # Toggle form visibility on button click

    
            if st.session_state.show_form:
                with st.form(key='add_knowledge_form'):
                    title = st.text_input("Titel", value=f"Hinweis zu folgender Frage: {user_query}")
                    content = st.text_area("Inhalt")
                    category = "User-Hinweis"
                    selected_german_tags = st.multiselect(
                        "Anwendbarkeit: Auf welche Personalkategorie ist das neue Wissen anwendbar? Bitte auswählen, mehrfache Auswahl ist erlaubt.",
                        list(set(tags_mapping.values())),
                        default=[
                            "Staatspersonal",
                            "Lehrperson VS",
                            "Lehrperson Sek II"
                        ]
                    )
                    submit_button = st.form_submit_button(label='Hinzufügen')
    
                    if submit_button and title and content:
                        # Convert the selected German tags to their corresponding English tags
                        selected_english_tags = []
                        for selected_german_tag in selected_german_tags:
                            selected_english_tags.extend(reverse_tags_mapping[selected_german_tag])
                        add_to_knowledge_base(title, content, category, selected_english_tags)
                        st.success("Neues Wissen erfolgreich hinzugefügt!")
    
            if 'delete_form' not in st.session_state:
                st.session_state.delete_form = False
        with col2:
            if st.button("Wissenselement löschen"):
                st.session_state.delete_form = not st.session_state.delete_form
                
            if st.session_state.delete_form:
                with st.form(key='delete_knowledge_form'):
                    # Just use the key directly instead of creating a tuple
                    entry_id_to_delete = st.selectbox(
                        "Wählen Sie das Wissenselement zum Löschen aus:", 
                        list(knowledge_base.keys()),
                        format_func=lambda x: f"{x}: {knowledge_base[x]['Title']}"
                    )
                    delete_button = st.form_submit_button(label='Löschen')
            
                    if delete_button and entry_id_to_delete:
                        delete_from_knowledge_base(entry_id_to_delete)
        # if st.session_state.delete_form:
            #     with st.form(key='delete_knowledge_form'):
            #         entry_id_to_delete = st.selectbox("Wählen Sie das Wissenselement zum Löschen aus:", [(key, knowledge_base[key]["Title"]) for key in knowledge_base.keys()])
            #         delete_button = st.form_submit_button(label='Löschen')
    
            #         if delete_button and entry_id_to_delete:
            #             delete_from_knowledge_base(entry_id_to_delete)
    
    
        st.write("")
        st.write("")
        st.write("")    
    
    # genAI-Teil
        if st.button("Mit Sprachmodell beantworten"):
            st.session_state.generating_answer = True  # Set this to true when button is clicked
        if st.session_state.get('generating_answer'):
            if user_query:
                st.session_state['generated_prompt'] = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)
                st.session_state['editable_prompt'] = st.text_area("**Prompt bearbeiten:**", st.session_state['generated_prompt'], height=300)
                if st.button("Promptengineering abgeschlossen") and st.session_state['editable_prompt']:
                    st.session_state.start_generating_answer = True  # Set this to true when button is clicked
                if st.session_state.get('start_generating_answer'):
                    
                    try:
                        # Handle Llama 3.1 model selection
                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                                {"role": "user", "content": st.session_state['editable_prompt']}
                            ],
                            model="llama-3.1-70b-versatile"
                        )
                        # Check if response is available
                        if chat_completion.choices and len(chat_completion.choices) > 0:
                            ai_message = chat_completion.choices[0].message.content
                            st.session_state['last_answer'] = ai_message
                            st.session_state['last_model'] = "Llama 3.1"

                        else:
                            st.warning("No response generated from Llama 3.1.")
        
                    # except groq.InternalServerError as e:
                    #     st.error(f"An internal server error occurred with the Groq API: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred with the Groq API: {str(e)}")
        
                    # Display the generated answer
                    if st.session_state['last_answer']:
                        st.subheader(f"Antwort SubSumary ({st.session_state['last_model']}):")
                        st.write(st.session_state['last_answer'])
                # else:
                #     st.warning("Please enter a query before generating an answer.")  # Warning for no query input
                    
                    # col1, col2 = st.columns(2)

                    
                    
                    # # with col1:
                    # #     if st.button("Mit GPT 4o beantworten"):
                    # #         if user_query:
                    # #            response = client.chat.completions.create(
                    # #                 model="gpt-4o",
                    # #                 messages=[
                    # #                     {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                    # #                     {"role": "user", "content": st.session_state['editable_prompt']}
                    # #                 ]
                    # #             )
                        
                    # #                 # Display the response from OpenAI
                    # #         if response.choices:
                    # #             ai_message = response.choices[0].message.content  # Corrected attribute access
                    # #             st.session_state['last_question'] = user_query
                    # #             st.session_state['last_answer_gpt4o'] = ai_message
                    # #     else:
                    # #         ai_message = st.session_state['last_answer_gpt4o']
                    # #     if st.session_state['last_answer_gpt4o']:
                    # #         st.subheader("Antwort subsumary:")
                    # #         st.write(st.session_state['last_answer_gpt4o'])
                
                    # with col2:
                    #     if st.button("Mit GPT 4o mini beantworten"):
                    #         if user_query:
                    #             query_vector = get_embeddings(user_query)
                    #             similarities = calculate_similarities(query_vector, article_embeddings)
                                
                    #             sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    #             filtered_articles = [(title, score) for title, score in sorted_articles if is_relevant_article(law_data[title], relevance)]
                    #             st.session_state.top_articles = filtered_articles[:10]
                    #             knowledge_similarities = calculate_similarities(query_vector, knowledge_base_embeddings)
                    #             st.session_state.top_knowledge_items = [(item_id, score) for item_id, score in sorted(knowledge_similarities.items(), key=lambda x: x[1], reverse=True) if is_relevant_article(knowledge_base[item_id], relevance)][:5]
                    #             prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)
                    #             response = client.chat.completions.create(
                    #                 model="gpt-4o-mini",
                    #                 messages=[
                    #                     {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                    #                     {"role": "user", "content": prompt}
                    #                 ]
                    #             )
                        
                    #                 # Display the response from OpenAI
                    #             if response.choices:
                    #                 ai_message = response.choices[0].message.content  # Corrected attribute access
                    #                 st.session_state['last_question'] = user_query
                    #                 st.session_state['last_answer'] = ai_message
                    #         else:
                    #             ai_message = st.session_state['last_answer']
                    #     if st.session_state['last_answer']:
                    #         st.subheader("Antwort subsumary:")
                    #         st.write(st.session_state['last_answer'])
                
                
        
                
                
            # if st.button("Prompt generieren und in die Zwischenablage kopieren"):
            #     if user_query and st.session_state.top_articles:
            #         # Generate the prompt
            #         prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)
            #         st.session_state['prompt'] = prompt
        
            #         # Create HTML with JavaScript to copy the prompt to the clipboard
            #         html_with_js = generate_html_with_js(prompt)
            #         st.components.v1.html(html_with_js)
        
            #         # Display the generated prompt in a text area
            #         st.text_area("Prompt:", prompt, height=300)
            #     else:
            #         if not st.session_state.top_articles:
            #             st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")
        
if __name__ == "__main__":
    main_app()

