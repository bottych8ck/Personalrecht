import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq  # Import the Groq client
from pydantic import BaseModel

class QueryAssessment(BaseModel):
    reformulate: bool
    new_query: str
    reason: str



# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)

load_dotenv()  # This line loads the variables from .env

openai_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

openai_client = openai.OpenAI(api_key=openai_api_key)
groq_client = Groq(api_key=groq_api_key)

def get_articles_details(top_articles, law_data):
    """
    Given a list of top articles (UIDs and scores), retrieves the title and content for each article.
    """
    articles_details = []

    for uid, score in top_articles:
        article_info = law_data.get(str(uid), None)
        if article_info:
            # Extract the title and content using the existing function
            title, all_paragraphs, _, _ = get_article_content(str(uid), law_data)

            # Combine title and content into a single dictionary entry
            articles_details.append({
                "title": title,
                "content": " ".join(all_paragraphs) if all_paragraphs else "" 
            })

    return articles_details

MAX_REFORMULATION_ATTEMPTS = 3  # Set a limit for the number of retries
def assess_and_reformulate_query_with_retries(user_query, top_articles, law_data, max_attempts=MAX_REFORMULATION_ATTEMPTS):
    
    current_attempt = 0
    reformulated_query = user_query

    while current_attempt < max_attempts:
        # Extract article details for each attempt
        articles_details = get_articles_details(top_articles, law_data)

        # Convert articles to a format suitable for model input
        articles_content_for_model = "\n".join(
            [f"Title: {article['title']}\nContent: {article['content']}" for article in articles_details]
        )

        # Create the structured completion request
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Use a model that supports structured outputs
            messages=[
                {"role": "system", "content": "You are an intelligent assistant that checks if a user query results in relevant articles. If not, you reformulate the query."},
                {"role": "user", "content": f"Original Query: {reformulated_query}\nTop Articles:\n{articles_content_for_model}"}
            ],
            response_format=QueryAssessment  # Specify the response format
        )

        # Extract and parse the structured output
        result = completion.choices[0].message.parsed

        if result.reformulate:
            # If reformulation is needed, update the query
            reformulated_query = result.new_query
            current_attempt += 1  # Increment the counter
            print(f"Reformulation attempt {current_attempt}: New Query - {reformulated_query}")
        else:
            # No further reformulation needed
            print("No reformulation needed.")
            break  # Exit the loop

    return reformulated_query




def get_embeddings(text):
    res = openai_client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    normalized_relevance = relevance.lower().replace("sek ii", "SEK II")
    tags = section_data.get("tags", [])
    normalized_tags = [tag.lower().replace("sek ii", "SEK II") for tag in tags]
    
    relevance_criteria = normalized_relevance
    
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

        content = " ".join(aggregated_content)
        prompt += f"\n{article_number}. §: {title} von folgendem Erlass: {name}\n"
        prompt += f"   - **Inhalt:** {content.strip()}\n"
        article_number += 1

    prompt += "\n"
    prompt += "Anfrage auf Deutsch beantworten. Prüfe die Anwendbarkeit der einzelnen § genau. Wenn ein Artikel keine einschlägigen Aussagen enthält, vergiss ihn.\n"
    return prompt

def main_app():

    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = ""
    if 'last_answer' not in st.session_state:
        st.session_state['last_answer'] = None
    if 'top_articles' not in st.session_state:
        st.session_state['top_articles'] = []
    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False
    if 'generating_answer' not in st.session_state:
        st.session_state['generating_answer'] = False

    if 'show_model_selection' not in st.session_state:
        st.session_state['show_model_selection'] = False  # Control flag for model selection visibility



    user_query = st.text_area("Hier Ihre Frage eingeben:", height=200, key="user_query_text_area")


    if st.button("Bearbeiten"):
        st.session_state['last_question'] = user_query
        st.session_state.generating_answer = False
        query_vector = get_embeddings(user_query)
        similarities = calculate_similarities(query_vector, article_embeddings)

        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.session_state.top_articles = sorted_articles[:10]
        st.session_state.submitted = True
        st.session_state.generating_answer = False  # Reset this when new query is processed
        final_query = assess_and_reformulate_query_with_retries(user_query, st.session_state['top_articles'], law_data)
    
        # Update the query in session state if it was reformulated
        if final_query != user_query:
            st.session_state['last_question'] = final_query
            st.success(f"Query reformulated: {final_query}")
        else:
            st.info("No reformulation needed.")
            if result.reformulate:
                # If reformulation is needed, update the query
                new_query = result.new_query
                st.session_state['last_question'] = new_query
                st.success(f"Query reformulated: {new_query}\nReason: {result.reason}")
            else:
                st.info("No reformulation needed.")

    if st.session_state.get('submitted'):
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
                # for item_id, _ in st.session_state.top_knowledge_items:
                #     item = knowledge_base.get(item_id, {})
                #     title = item.get("Title", "Unbekannt")
                #     content = item.get("Content", "")
                #     year = item.get("Year", "")
                    
                #     if isinstance(content, list):
                #         # Join list into a single string with double spaces at the end of each line to ensure Markdown respects line breaks
                #         content = '  \n'.join(content)
                    
                #     # Display the title and content with proper formatting
                #     st.markdown(f"**{title}**")
                #     st.markdown(f"*Auskunft aus dem Jahr {year}*")  # Display the year under the title
                #     st.markdown(content)




        st.write("")
        st.write("")
        st.write("")   
        st.write("")    


        col1, col2 = st.columns(2)
                         
        with col1:
            if st.button("Antwort mit Sprachmodell generieren"):
                st.session_state.generating_answer = True  # Set this to true when button is clicked
            if st.session_state.get('generating_answer'):
                if user_query:  # Check if a user query is entered
                    prompt = generate_prompt(user_query, st.session_state.top_articles, law_data)
                    st.write("Sending request to Groq API...")
                    try:
                        # Handle Llama 3.1 model selection
                        chat_completion = groq_client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama-3.1-70b-versatile"
                        )
                        st.write("Received response from Groq API:", chat_completion)
                        # Check if response is available
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
        
                    # Display the generated answer
                    if st.session_state['last_answer']:
                        st.subheader(f"Antwort summary ({st.session_state['last_model']}):")
                        st.write(st.session_state['last_answer'])
                else:
                    st.warning("Please enter a query before generating an answer.")  # Warning for no query input
    
            
        with col2:
                
            if st.button("Prompt generieren und in die Zwischenablage kopieren"):
                if user_query and st.session_state.top_articles:
                    # Generate the prompt
                    prompt = generate_prompt(user_query, st.session_state.top_articles, law_data)
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
    main_app()  # Correctly call the main application function

