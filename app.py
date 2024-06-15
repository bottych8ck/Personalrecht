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

with open('knowledge_base_embeddings.json', 'r') as file:
    knowledge_base_embeddings = json.load(file)

with open('knowledge_base.json', 'r') as file:
    knowledge_base = json.load(file)
load_dotenv()  # This line loads the variables from .env
logo_path = 'subsumary_Logo_1farbig_schwarz.png'

api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)

def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
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

# def generate_html_with_js(prompt):
#     return f"""
#     <textarea id='text_area' style='opacity: 0; position: absolute; left: -9999px;'>{prompt}</textarea>
#     <button onclick='copyToClipboard()'>Text in die Zwischenablage kopieren</button>
#     <script>
#     function copyToClipboard() {{
#         var copyText = document.getElementById('text_area');
#         copyText.style.opacity = 1; // Make the textarea visible to enable selection
#         copyText.select();
#         document.execCommand('copy');
#         copyText.style.opacity = 0; // Hide the textarea again
#         alert('Copied to clipboard!');
#     }}
#     </script>
#     """

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
    
    # prompt += "\nZusätzlich berücksichtige folgende allgemeine Grundsätze und Prinzipien:\n"
    # for item_id, _ in top_knowledge_items:
    #     item = knowledge_base.get(item_id, {})
    #     title = item.get("Title", "Unbekannt")
    #     content = ' '.join(item.get("Content", []))
    #     prompt += f"- {title}: {content}\n"

   
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
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ""
    if 'top_knowledge_items' not in st.session_state:
        st.session_state.top_knowledge_items = [] 

    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200)

    relevance_options = ["Staatspersonal", "Lehrperson VS", "Lehrperson Sek II"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Staatspersonal, Lehrpersonen der Volksschule oder Lehrpersonen der Berufsfach- und Mittelschulen bezieht:", relevance_options)

    if 'top_articles' not in st.session_state:
        st.session_state.top_articles = []
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if user_query != st.session_state['last_question']:
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)
        
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        filtered_articles = [(title, score) for title, score in sorted_articles if is_relevant_article(law_data[title], relevance)]
        st.session_state.top_articles = filtered_articles[:10]
        knowledge_similarities = calculate_similarities(query_vector, knowledge_base_embeddings)
        st.session_state.top_knowledge_items = [(item_id, score) for item_id, score in sorted(knowledge_similarities.items(), key=lambda x: x[1], reverse=True) if is_relevant_article(knowledge_base[item_id], relevance)][:5]
    if st.button("Relevante Bestimmungen"):
        st.session_state.submitted = True
        st.write("Die folgenden Bestimmungen und Hinweise passen am Besten auf die Anfrage.")
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
    col1, col2 = st.columns(2)
                
    with col1:
        if st.button("Mit GPT 4o beantworten") and user_query:
            
            if user_query != st.session_state['last_question']:
                query_vector = get_embeddings(user_query)
                prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)
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
            st.subheader("Antwort subsumary:")
            st.write(st.session_state['last_answer'])
        else:
            st.warning("Bitte geben Sie eine Anfrage ein.")
            

    with col2:
        if st.session_state.submitted:
            if st.button("Prompt generieren und in die Zwischenablage kopieren"):
                if user_query and st.session_state.top_articles:
                    # Generate the prompt
                    prompt = generate_prompt(user_query, relevance, st.session_state.top_articles, law_data, st.session_state.top_knowledge_items)
                    st.session_state['prompt'] = prompt
        
                    # Create HTML with JavaScript to copy the prompt to the clipboard
                    html_with_js = generate_html_with_js(prompt)
                    html(html_with_js)
        
                    # Display the generated prompt in a text area
                    st.text_area("Prompt:", prompt, height=300)
                else:
                    if not user_query:
                        st.warning("Bitte geben Sie eine Anfrage ein.")
                    if not st.session_state.top_articles:
                        st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")



def main():
    main_app()

if __name__ == "__main__":
    main()
