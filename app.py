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



User
PS Y:\DEK\Allgemein\Rechtsdienst\Zwischenspeicher\Programmieren\Lawquery with embeddings\StWG> pip install openai
pip : Die Benennung "pip" wurde nicht als Name eines Cmdlet, einer Funktion, einer Skriptdatei oder eines ausführbaren 
Programms erkannt. Überprüfen Sie die Schreibweise des Namens, oder ob der Pfad korrekt ist (sofern enthalten), und 
wiederholen Sie den Vorgang.
In Zeile:1 Zeichen:1
+ pip install openai
+ ~~~
    + CategoryInfo          : ObjectNotFound: (pip:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
 
PS Y:\DEK\Allgemein\Rechtsdienst\Zwischenspeicher\Programmieren\Lawquery with embeddings\StWG>    
ChatGPT
It appears that you're encountering an issue where PowerShell doesn't recognize the pip command, which is typically used to install Python packages like openai. This issue is often due to Python not being properly added to your system's PATH environment variable, or pip not being installed correctly with Python. Here are some steps you can take to resolve this issue:

Check if Python is Installed: First, verify that Python is installed on your system. You can do this by typing python --version or python3 --version in PowerShell. If Python is installed, it should return the version number.

Verify Python Installation Path: If Python is installed, find out where it is installed. You can usually find Python in directories like C:\Python39 (the version number may vary).

Add Python to PATH:

If Python is installed but not in the PATH, you'll need to add it. You can do this by searching for 'Environment Variables' in Windows and editing the PATH variable to include the path to your Python installation and the Scripts folder (e.g., C:\Python39\;C:\Python39\Scripts\;).
After modifying the PATH, restart PowerShell and try the pip command again.
Install/Upgrade Python with pip:

If Python is not installed, download and install it from the official Python website. Ensure you check the option to 'Add Python to PATH' during installation.
If you have an older version of Python without pip, consider upgrading to a newer version.
Use Python's -m Flag: If pip is installed but not recognized, you can try using it with Python's -m flag. For example, use python -m pip install openai or python3 -m pip install openai.

Check for Multiple Python Installations: If you have multiple versions of Python installed, it can sometimes cause conflicts. Ensure you're using the correct version of Python and pip that you intend to use.

Using PowerShell as Administrator: Sometimes, running PowerShell as an administrator can help with permission-related issues.

If after trying these steps, you're still encountering issues, it could be helpful to look up the specific error messages or consider reinstalling Python, ensuring that pip is included in the installation.

User
it doesent work, its my stupid work computer, and I have already talked to the support and they added the necessary envioronment variables
ChatGPT
If you're still facing issues with pip on your work computer even after the support team has added the necessary environment variables, there could be a few additional steps or workarounds to consider:

Check for Restrictions: Sometimes, corporate policies or network restrictions can limit the ability to install software or modify certain system settings. It's possible that such restrictions could be causing the issue with pip. If you suspect this might be the case, you may need to speak with your IT department for further assistance.

Alternative Installation Methods:

Python Embeddable Zip File: For environments where you can't install Python traditionally, you might be able to use the Python embeddable zip file. This is a portable version of Python that can be run from any directory without installation. You can download it from the Python Releases for Windows page.
Virtual Environments: If Python is installed but pip isn't working, you can try using Python's built-in venv module to create a virtual environment. This can sometimes bypass path and permission issues. You can create a virtual environment with python -m venv myenv and then activate it.
Use a Portable Python Distribution:

WinPython or Portable Python: These are portable distributions of Python for Windows. They are designed to be run from a USB drive or other portable storage without needing to be installed on the system.
Manual Installation of Packages:

If you can't use pip, you can manually download the package files (wheel files, usually with a .whl extension) from a site like PyPI or GitHub and install them manually using Python. This is more complex and less convenient than using pip, but it can be a workaround if pip is not available.
Use an Online Python Environment:

If all else fails and you just need to run some Python code, consider using an online Python interpreter or environment. Websites like Replit, Google Colab, or Jupyter Notebook can provide a Python environment in your browser without the need to install anything on your computer.
Consult with IT Support Again:

If you're still experiencing issues, it may be worth consulting with your IT support team again. They may have additional insights or be able to provide alternative solutions specific to your corporate environment.
Remember, if you're working within a corporate or restricted IT environment, always follow your organization's policies and guidelines when attempting to resolve software issues.

User
how can I create a venv? via terminal?
ChatGPT
Yes, you can create a virtual environment (venv) in Python using the terminal. Here's a step-by-step guide to do this:

Open Your Terminal or Command Prompt:

On Windows, you can use PowerShell or Command Prompt.
On macOS or Linux, you can use the Terminal.
Navigate to Your Project Directory:

Use the cd command to navigate to the directory where you want to create the virtual environment. For example:
bash
Copy code
cd path\to\your\project
Create the Virtual Environment:

Run the following command to create a virtual environment. Replace env_name with the name you want to give to your virtual environment:
Copy code
python -m venv env_name
This will create a directory named env_name in your current directory, containing the virtual environment.
Activate the Virtual Environment:

The activation command differs based on your operating system:
On Windows (Command Prompt):
Copy code
.\env_name\Scripts\activate
On Windows (PowerShell):
mathematica
Copy code
.\env_name\Scripts\Activate.ps1
On macOS or Linux:
bash
Copy code
source env_name/bin/activate
Once activated, you should see the name of your virtual environment (in this case, env_name) in the prompt. This indicates that the virtual environment is active.
Using the Virtual Environment:

While the virtual environment is active, any Python packages you install using pip will be installed in the virtual environment, not globally.
You can now install packages specific to your project without affecting the global Python installation.
Deactivate the Virtual Environment:

When you're done, you can deactivate the virtual environment by running:
Copy code
deactivate
This will return you to your system’s default Python environment.
Remember, each time you start a new terminal session and want to work on your project, you need to activate the virtual environment. This ensures that you're using the correct Python interpreter and dependencies for your project.

User
nothing works. its a disaster but I dont want to waste time on this. My plan is to run a streamlit app. I have some google colab code, but I know, this has to be changed in order to run it. I could just save the files in github and run the app from there, correct? 
ChatGPT
If you're facing challenges setting up a Python environment on your work computer, using Google Colab and GitHub in conjunction with Streamlit offers a good workaround. Here's a step-by-step guide on how you can achieve this:

Prepare Your Code in Google Colab:

Start by finalizing your code in Google Colab. Since Colab notebooks are interactive, ensure that your code can run as a script, which is necessary for Streamlit.
You may need to modify the code to remove interactive elements like input prompts, as these won't work the same way in a Streamlit app.
Save Your Notebook as a Python Script:

In Google Colab, you can download your notebook as a Python script. Go to File → Download → Download .py.
This will give you a .py file with your Colab code, which you can use as the basis for your Streamlit app.
Create a GitHub Repository:

If you don’t already have a GitHub account, create one at github.com.
Create a new repository and upload your Python script there. You can also include any necessary data files or requirements.txt file for Python dependencies.
Modify Your Code for Streamlit:

Streamlit uses a simple API to create web apps. You will likely need to adapt your code to use Streamlit's functions for user input and data display.
For example, instead of using print() for output, you would use Streamlit's st.write() function.
Check out the Streamlit documentation for guidance on how to adapt your code.
Deploy Your Streamlit App:

Streamlit offers Streamlit Sharing, a platform where you can deploy your apps directly from your GitHub repository.
Go to Streamlit Sharing and sign up if you haven't already.
Connect your GitHub repository and select the main script file to deploy your app.
Streamlit will provide you with a URL where your app is hosted and can be accessed.
Run and Share Your App:

Once your app is deployed, you can run it through the provided URL.
Share the URL with others so they can view and interact with your Streamlit app.
Using Google Colab for development and GitHub in conjunction with Streamlit for deployment is a great way to bypass local environment issues. This approach leverages cloud-based services, making it easier to develop and share your applications.

User
but in google colab I canot simply download a .py file, it seems. or can i? I dont have that option in the files meneu
ChatGPT
You're right that Google Colab doesn't directly provide an option in the File menu to download your notebook as a .py script. However, there is a straightforward workaround to convert your Colab notebook to a Python script:

Convert Notebook to Python Script via Colab:

Open your notebook in Google Colab.
Go to File in the top menu.
Then go to Download → Download .ipynb to download the notebook file to your computer.
Convert .ipynb to .py Locally:

Once you have the .ipynb file, you can convert it to a .py file using a tool like Jupyter Notebook or Jupyter Lab installed on your local machine. If you don't have Jupyter installed, you can install it using pip:
Copy code
pip install notebook
After installing Jupyter, use the following command in the terminal to convert the .ipynb file to a .py file:
css
Copy code
jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
Replace [YOUR_NOTEBOOK] with the name of your notebook file. This will create a .py file with the same name as your notebook.
Alternative Online Tools:

If you don't want to install Jupyter, there are also online tools available that can convert .ipynb files to .py. You can upload your .ipynb file to these tools, and they will handle the conversion for you.
Editing the Python Script for Streamlit:

Once you have your Python script, you can modify it to be compatible with Streamlit, as discussed earlier.
By converting your Colab notebook to a Python script, you'll be able to adapt it for a Streamlit app and proceed with deployment.

User
what are these tools?
ChatGPT
There are several online tools available that can convert Jupyter Notebooks (.ipynb files) to Python scripts (.py files). Here are a few you can use:

Jupyter Notebook Converters:

Websites like Jupyter Notebook Convert (found at nbconvert.jupyter.org) allow you to upload a .ipynb file and download it as a .py file.
These converters are straightforward to use: you just need to upload your notebook file and select the format you want to convert it to.
Online IDEs:

Online integrated development environments (IDEs) like Replit (replit.com) can import Jupyter Notebooks. Once imported, you can copy the code into a Python script within the platform.
Some of these IDEs might require you to sign up for an account.
Cloud-Based Jupyter Services:

Services like Google Colab itself or IBM Watson Studio can open .ipynb files. After opening the file, you can manually copy the code into a Python script file.
File Conversion Services:

General file conversion websites sometimes offer the ability to convert between different coding file formats, including .ipynb to .py. However, their availability and reliability can vary.
GitHub Gists:

If you upload your .ipynb file to GitHub as a Gist, GitHub renders the notebook. While it doesn't directly offer a .py download, you can easily copy the code from the rendered view to a new .py file.
Remember, when using online tools, especially those requiring uploads, it's important to be mindful of the sensitivity and privacy of your data. Avoid using these services for confidential or proprietary information. For general-purpose code conversion where data sensitivity is not an issue, these tools can be quite convenient.

User
but I dont understand, I can just copy my code into a py-file? 
ChatGPT
Yes, you can directly copy the code from a Jupyter Notebook (like one you've used in Google Colab) into a .py file. This is a simple and effective way to transfer your code for use in a different environment, such as for creating a Streamlit app. Here’s how you can do it:

Open Your Notebook in Google Colab:

Go to your notebook in Google Colab.
Copy the Code:

For each cell in your notebook that contains code you want to use, simply copy the code.
You can ignore cells that are not relevant or that contain only Markdown text or outputs.
Paste into a New .py File:

Open a text editor (like Notepad, Notepad++, VSCode, or any other code editor).
Paste the copied code into a new file in your text editor.
You can structure the code as needed, ensuring it follows a logical and executable sequence as a script.
Save the File as .py:

Save the file with a .py extension, for example, my_script.py.
This file is now a Python script that can be run like any other Python script.
Adjustments for Streamlit:

If you are adapting this script for a Streamlit app, you will need to modify parts of it to use Streamlit’s functions and widgets for user interaction.
Replace print statements, input functions, and other interactive elements with Streamlit’s counterparts like st.write(), st.button(), st.text_input(), etc.
Run and Test:

Once your script is ready and adapted for Streamlit, you can run it locally (if you have Streamlit installed) or deploy it as a Streamlit app.
This method of copying code is straightforward and works well when you have a clear understanding of which parts of your notebook are necessary for your script and how to structure a Python script. Remember, a Jupyter Notebook often contains cells that are not strictly sequential or that include interactive elements not suitable for a script format, so some adaptation might be necessary.

User
This is the code I want to have run in a streamlit app. It needs to be changed a bit obviously. The two dictionarys embeddings_dict and lawcontent_dict will be uploaded to github by me. Here is the code that needs to be changed. 

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
ChatGPT
To adapt your Python function for a Streamlit app, you'll need to make some changes to handle user input and display the output in the Streamlit interface. Here's an adapted version of your function for a basic Streamlit app:

python
Copy code
import streamlit as st

# Assuming embeddings_dict and lawcontent_dict are loaded from GitHub or elsewhere
# embeddings_dict = ...
# lawcontent_dict = ...

def generate_prompt(user_query, relevance, top_articles, lawcontent_dict):
    # Your existing function code remains largely the same
    # ...

# Streamlit app
def main():
    st.title("Legal Query Answering System")

    # User input for the query
    user_query = st.text_input("Enter your query:")

    # User selection for relevance
    relevance_options = ["assembly", "mail voting", "none"]
    relevance = st.selectbox("Select relevance:", relevance_options)

    # Assuming you have a way to determine top_articles based on the query
    # For demo purposes, let's assume it's a static list for now
    top_articles = [("Article1", 0.95), ("Article2", 0.90)]  # Example

    # Button to generate prompt
    if st.button("Generate Prompt"):
        if user_query:  # Ensure there is a query to process
            prompt = generate_prompt(user_query, relevance, top_articles, lawcontent_dict)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
Explanation of Changes:

User Input: The user's query and the relevance are captured using Streamlit's text_input and selectbox widgets.

Generating Prompt: A button is added using st.button. When clicked, it calls the generate_prompt function with the user's input.

Displaying Output: The output of the function (the generated prompt) is displayed in a text area using st.text_area.

Top Articles: In your original function, top_articles seems to be a parameter. In this Streamlit version, you'll need to determine how to populate this. For demonstration, I've used a static list, but in practice, this might come from another part of your code that processes the user's query.

Running the App: To run this app, save the code in a Python file (e.g., app.py) and run it using streamlit run app.py in your terminal.

Remember, this is a basic adaptation. Depending on your full application's complexity and how top_articles is determined, you might need additional code or adjustments.


2 / 2

User
ah shuut, I gave you only a part of the code, you are correct, sorry: here is the full thing:
client = openai.OpenAI(api_key='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')
# Mapping for relevance choices
relevance_phrases = {
    "assembly": "Die Frage betrifft Gemeindeversammlungen.",
    "mail voting": "Die Frage betrifft Urnenwahl."
}

# Prompt for relevance
relevance = input("Is the question related to 'Assembly', 'Mail Voting', or 'None'? ").strip().lower()

# Original user query
user_query = "Wieviele Tage vor der Gemeindeversammlung muss ich die Botschaft abschicken?"

def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

# Enhance the query based on the relevance choice
enhanced_user_query = user_query

# Enhance the query based on the relevance choice
if relevance in relevance_phrases:
    enhanced_user_query+= " " + relevance_phrases[relevance]

# Get embeddings for the enhanced query
query_vector = get_embeddings(enhanced_user_query)


def is_relevant_article(section_data, relevance):
    tags = section_data.get("tags", [])
    if relevance == 'assembly':
        return any("Assembly" in tag for tag in tags)
    elif relevance == 'mail voting':
        return any("Mail Voting" in tag for tag in tags)
    else:  # If relevance is 'none' or any other value, consider all articles
        return True

def get_relevant_articles(lawcontent_dict, relevance):
    relevant_articles = {}
    for section, section_data in lawcontent_dict.items():
        if is_relevant_article(section_data, relevance):
            relevant_articles[section] = section_data
    return relevant_articles


def calculate_similarities(query_vector, embeddings_dict):
    query_vector = np.array(query_vector).reshape(1, -1)
    similarities = {}

    for title, article_vector in embeddings_dict.items():
        article_vector = np.array(article_vector).reshape(1, -1)
        similarity = cosine_similarity(query_vector, article_vector)[0][0]
        similarities[title] = similarity

    return similarities


# Filter the articles based on the specified relevance
relevant_lawcontent_dict = get_relevant_articles(lawcontent_dict, relevance)

# Calculate similarities only for relevant articles
similarities = calculate_similarities(query_vector, {title: embeddings_dict[title] for title in relevant_lawcontent_dict if title in embeddings_dict})

# Sort articles by similarity
sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
# Print the top 5 most relevant articles
for title, score in sorted_articles[:15]:
    print(f"{title}: {score}")
top_articles = sorted_articles[:5]

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



prompt = generate_prompt(user_query, relevance, top_articles, lawcontent_dict)
print(prompt)
ChatGPT
To adapt your full code for a Streamlit app, there are several modifications to be made. This includes handling user inputs via Streamlit widgets, displaying outputs in the app, and reorganizing the code flow to fit a web app context.

Here's an adapted version of your code for a Streamlit app:

python
Copy code
import streamlit as st
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming embeddings_dict and lawcontent_dict are loaded from GitHub or elsewhere
# embeddings_dict = ...
# lawcontent_dict = ...

# Initialize OpenAI client
client = openai.OpenAI(api_key='your_api_key_here')

def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def is_relevant_article(section_data, relevance):
    # Your existing code
    # ...

def get_relevant_articles(lawcontent_dict, relevance):
    # Your existing code
    # ...

def calculate_similarities(query_vector, embeddings_dict):
    # Your existing code
    # ...

def generate_prompt(user_query, relevance, top_articles, lawcontent_dict):
    # Your existing code
    # ...

# Streamlit app
def main():
    st.title("Legal Query Answering System")

    # User inputs
    user_query = st.text_input("Enter your query:")
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
Key Adaptations:

User Input via Streamlit: The user's query and relevance are taken via Streamlit's text_input and selectbox.

Processing Logic: When the user clicks the "Generate Prompt" button, the code processes the query, calculates similarities, and generates a prompt.

Displaying Output: The generated prompt is displayed using st.text_area.

API Key Handling: Be cautious with your OpenAI API key. It's typically not a good practice to hardcode API keys, especially in public repositories. Consider using environment variables or Streamlit's secrets management for this purpose.

This adapted code should provide a good foundation for your Streamlit app. Remember to test and adjust it based on the specific needs and structure of your application.

User
great, regarding the dictionary, I dont want the hardcoded into the script, because they are huge. But in another project, I could just upload them to github and that worked in the streamlit app
ChatGPT
Using large dictionaries directly in your Streamlit app script can indeed be impractical, especially if they are very large. Uploading them to GitHub and then fetching them into your Streamlit app is a good approach. Here’s how you can do that:

Upload Your Dictionaries to GitHub:

Convert your dictionaries (embeddings_dict and lawcontent_dict) to a suitable file format, such as JSON.
Upload these files to a GitHub repository.
Access Files in Your Streamlit App:

You can use Python's requests library to fetch the files from GitHub.
If your repository is public, you can access the raw file URL directly. If it's private, you might need to use GitHub's API and handle authentication.
Here's an example of how you can modify your Streamlit app to load these dictionaries from GitHub:

python
Copy code
import streamlit as st
import requests
import json

# Function to load a dictionary from a GitHub raw file URL
def load_dict_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        st.error(f"Failed to load data from {url}")
        return {}

# URLs of the raw files on GitHub
embeddings_dict_url = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/embeddings_dict.json'
lawcontent_dict_url = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/lawcontent_dict.json'

# Load the dictionaries
embeddings_dict = load_dict_from_github(embeddings_dict_url)
lawcontent_dict = load_dict_from_github(lawcontent_dict_url)

# Rest of your Streamlit app code...
Remember to replace the URLs with the actual URLs of your raw files on GitHub. This approach will fetch the dictionaries when the app is started or refreshed.

Note: Fetching large files from GitHub every time the app starts might slow down the initial loading of your app. If the dictionaries don't change often, you might consider caching them using Streamlit’s @st.cache decorator to improve performance.

User
I have something like this in another app:
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

def calculate_similarities(query_vector, article_embeddings):
    query_vector = np.array(query_vector).reshape(1, -1)
    similarities = {}

    for title, article_vector in article_embeddings.items():
ChatGPT
In your existing Streamlit app, you're loading JSON data from local files and using an environment variable for the OpenAI API key. This is a good approach, especially for local development or when deploying in an environment where you have control over the file system and environment variables.

Here's a brief overview of the key components in your code:

Loading Data from JSON Files:

You're using with open('filename.json', 'r') as file: to open and read JSON files. This is a standard way to load data in Python and works well for local or server-based deployments where the file paths are known and accessible.
Environment Variables with dotenv:

The load_dotenv() function from the dotenv package loads environment variables from a .env file. This is a secure way to manage sensitive data like API keys, especially during development.
os.getenv('OPENAI_API_KEY') fetches the OpenAI API key set in your .env file.
Fetching Embeddings and Calculating Similarities:

The get_embeddings function communicates with the OpenAI API to get embeddings for given text.
The calculate_similarities function calculates similarities between query vectors and article embeddings, which seems to be part of a larger functionality for retrieving relevant information based on the query.
For deploying your Streamlit app with external data files (article_embeddings.json and law_data.json):

If deploying on a platform where you can upload files (like Streamlit Sharing), you can upload these JSON files along with your app.
If the files are too large or you prefer not to upload them directly, you can host them externally (like on GitHub, as discussed previously) and modify your code to fetch them remotely.
Regarding the environment variable for the API key:

If deploying on Streamlit Sharing, you can use Streamlit's secret management to securely store your API key.
Ensure the .env file or the API key is never uploaded to public repositories for security reasons.
By adapting your app based on where and how it's deployed, you can maintain both functionality and security, whether running locally or in a cloud-based environment.

User
jI dont think loading the JSON Data is only possible in a local env, because I have those files in my github-data und run the app via github..
ChatGPT
You are correct. If you have the JSON data files stored in a GitHub repository and your Streamlit app is set up to fetch these files from there, then you can certainly run your app using these remote data sources. This setup is particularly useful when you want to deploy your app without having to upload large data files directly to the deployment platform.

Here's how you can adapt your Streamlit app to load JSON data from GitHub:

Store Your JSON Files on GitHub:

Make sure your article_embeddings.json and law_data.json files are available in your GitHub repository.
Fetch Files from GitHub in Your Streamlit App:

Use Python's requests library to fetch the files from their GitHub raw content URLs.
Ensure the files in your GitHub repository are publicly accessible if your app needs to fetch them without authentication.
Here is a modified version of your file-loading code to fetch data from GitHub:

python
Copy code
import requests
import json

def load_json_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        raise Exception(f"Failed to load data from {url}")

# URLs of the raw JSON files on GitHub
article_embeddings_url = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/article_embeddings.json'
law_data_url = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/law_data.json'

# Load the data
article_embeddings = load_json_from_github(article_embeddings_url)
law_data = load_json_from_github(law_data_url)

# Rest of your Streamlit app code...
Replace the URLs with the actual URLs to the raw JSON files in your GitHub repository. This code will dynamically fetch the data when your Streamlit app runs, ensuring it's always up-to-date with what's in your GitHub repository.

This approach provides flexibility for your app's deployment and makes it easier to update the data without redeploying the app—simply update the files in your GitHub repository.

User
what does that: Für-Stimm-und-Gemeinderecht had recent pushes 19 minutes ago

mean in github?
ChatGPT
The message "Für-Stimm-und-Gemeinderecht had recent pushes 19 minutes ago" in GitHub refers to recent activity in a GitHub repository. Let me break down the message for you:

"Für-Stimm-und-Gemeinderecht": This is likely the name of a GitHub repository. Repository names in GitHub are often used to identify different projects or collections of code and files.

"had recent pushes": In the context of GitHub, a "push" refers to the action of uploading local changes to a remote repository. When someone makes changes to their local copy of a repository (such as modifying files, adding new files, or fixing bugs), they commit those changes and then "push" them to the repository hosted on GitHub. This makes the changes available to others and updates the repository.

"19 minutes ago": This is a timestamp indicating when the last push to the repository occurred. In this case, it means that someone pushed changes to the "Für-Stimm-und-Gemeinderecht" repository approximately 19 minutes before the time you saw this message.

Overall, the message indicates that the "Für-Stimm-und-Gemeinderecht" repository has been actively updated with new changes or contributions very recently. This is a common sight on GitHub, especially in active projects where multiple contributors are continuously working and updating the codebase.

User
so this is a separate branch of a repo. can I add specific files just for that branch?
ChatGPT
Yes, you can add specific files to a separate branch in a Git repository without affecting other branches. Each branch in Git is like an independent line of development, allowing you to work on different features or versions of your project simultaneously. Here's how you can do this:

Switch to the Specific Branch:

First, make sure you're on the branch where you want to add the files. You can switch to your desired branch using the command:
bash
Copy code
git checkout branch-name
Replace branch-name with the name of the branch you want to work on.
Add Your Files to the Branch:

Add the files you want to include in this branch to your project folder.
Then, stage these files for commit using:
bash
Copy code
git add file1 file2
Replace file1 file2 with the names of the files you've added. You can also use git add . to stage all new and modified files.
Commit the Changes:

Once you've staged the new files, commit them to the branch with a message describing your changes:
bash
Copy code
git commit -m "Added specific files to the branch"
Push the Changes to the Remote Repository:

After committing the changes locally, push them to the remote repository:
bash
Copy code
git push origin branch-name
This will upload your changes to the specified branch on the remote repository (like GitHub).
Verify the Changes:

After pushing, you can go to GitHub and check the branch to ensure your files are there. Remember, these changes will be isolated to the branch you're working on and won't affect other branches unless you merge them.
If the branch you want to work on doesn't exist yet, you can create it using git checkout -b new-branch-name. This command creates a new branch and switches to it.

Remember, it's a good practice to keep your branches focused on specific features or purposes to avoid conflicts and maintain a clear project history.

User
oke, I did the changes. They are in a separate branch. Can I deploy this branch to streamlit? 
ChatGPT
Yes, you can deploy a specific branch of your GitHub repository to Streamlit. When setting up or updating your Streamlit app, you have the option to specify which branch you want to deploy. Here's how you can do it:

Go to Streamlit Share:

Visit Streamlit Sharing and log in with your GitHub account.
Deploy a New App or Update an Existing One:

If you're deploying a new app, click on "New app" and then select your GitHub repository.
If you're updating an existing app, find your app in the Streamlit dashboard and select "Edit".
Select the Branch:

During the setup process, you'll have the option to select the branch you want to deploy.
Choose the branch that contains the changes you want to deploy.
Complete the Deployment:

After selecting the branch, complete any other necessary setup steps (like adding environment variables, if needed).
Once everything is set up, click "Deploy" to start the deployment process.
Check Your Deployed App:

Streamlit will build and deploy your app based on the selected branch. You can monitor the deployment process in the Streamlit dashboard.
When the deployment is complete, Streamlit will provide a URL to access your app.
Managing Branches for Deployment:

Keep in mind that if you continue to make changes in different branches, you'll need to manually update the deployed branch in Streamlit if you want those changes to be reflected in your live app.
By following these steps, you can ensure that the specific branch you want to showcase is the one that's deployed in your Streamlit app. This allows you to have different versions of your app (like development, testing, or production versions) deployed from different branches as needed.


2 / 2

User
can I change the name of a branch?
ChatGPT
Yes, you can change the name of a branch in Git, both locally and on a remote repository like GitHub. Here are the steps to rename a branch:

Renaming a Branch Locally
Switch to the Branch You Want to Rename:

bash
Copy code
git checkout old-branch-name
Replace old-branch-name with the current name of the branch.

Rename the Branch:

bash
Copy code
git branch -m new-branch-name
Replace new-branch-name with the new name you want for the branch.

Updating the Renamed Branch on a Remote Repository
If the branch you renamed is also on a remote repository (like GitHub), you'll need to update it there as well:

Push the Renamed Branch and Reset the Upstream Reference:

bash
Copy code
git push origin -u new-branch-name
Delete the Old Branch from the Remote Repository:

If the old branch name is on the remote, you should delete it to avoid confusion:
bash
Copy code
git push origin --delete old-branch-name
Considerations
Check for Open Pull Requests: If there are any open pull requests targeting the old branch name, they will need to be updated to the new branch name.
Inform Your Team: If you are working in a team, inform everyone about the branch name change, as it will affect anyone else who is working with the same branch.
Update CI/CD Pipelines: If you have continuous integration or continuous deployment set up, you may need to update your configurations to reflect the new branch name.
Remember, renaming a branch is a significant change, especially in a collaborative environment. It's important to coordinate such changes with your team to avoid any disruptions in the workflow.

User
can I change a separate branch into an own repo?
ChatGPT
Yes, you can convert a branch from an existing repository into a separate, standalone repository. This process is particularly useful when a branch has evolved to have a distinct purpose or functionality separate from the main project. Here’s how you can do it:

Method 1: Cloning and Changing the Remote URL
Clone the Branch:

First, clone the specific branch from your existing repository:
bash
Copy code
git clone -b branch-name --single-branch https://github.com/username/repository.git new-repository
Replace branch-name, username, and repository with the respective branch name and original repository details. new-repository is the name of the directory for your new repository.
Create a New Repository on GitHub:

Go to GitHub and create a new repository. This will be the destination for your branch-turned-repository.
Change the Remote URL:

Navigate into the cloned directory (new-repository) and change the remote URL to the new repository:
bash
Copy code
cd new-repository
git remote set-url origin https://github.com/username/new-repository.git
Replace username and new-repository.git with your GitHub username and the new repository name.
Push to the New Repository:

Finally, push the code to the new repository:
bash
Copy code
git push -u origin master
Method 2: Using git subtree
The git subtree command can be used to split a branch into a new repository, preserving its commit history.

Split the Branch:

In the original repository, use the git subtree command to split the branch:
bash
Copy code
git subtree split --prefix=path/to/your/subfolder -b new-branch
This creates a new branch (new-branch) containing the history of the subfolder.
Create a New Repository on GitHub and clone it to your local system.

Pull the Split Branch into the New Repository:

In the new repository directory, pull the split branch:
bash
Copy code
git pull /path/to/original/repo new-branch
Push the Changes:

Push the changes to the new repository’s remote:
bash
Copy code
git push origin master
Considerations
Dependencies and Project Structure: If your branch has dependencies or references to other parts of the original repository, you'll need to adjust them accordingly in the new repository.
Collaboration Impact: If others are working on the same branch, inform them about the change to coordinate further development.
Continuous Integration/Deployment: If you have any CI/CD pipelines set up, they will need to be reconfigured for the new repository.
Both methods will effectively create a new repository based on a branch of an existing repository. The best method depends on your specific requirements and whether you need the entire commit history of the branch.

User
oke, I deployed it, but got an error:
The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

2023-12-13 15:20:59.341 503 GET /script-health-check (10.12.26.248) 381.64ms

2023-12-13 15:21:04.109 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 19, in <module>

    client = openai.OpenAI(api_key=api_key)

  File "/home/adminuser/venv/lib/python3.9/site-packages/openai/_client.py", line 92, in __init__

    raise OpenAIError(

openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

2023-12-13 15:21:04.345 503 GET /script-health-check (10.12.26.248) 309.22ms

2023-12-13 15:21:09.041 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 19, in <module>

    client = openai.OpenAI(api_key=api_key)

  File "/home/adminuser/venv/lib/python3.9/site-packages/openai/_client.py", line 92, in __init__

    raise OpenAIError(

openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

2023-12-13 15:21:09.263 503 GET /script-health-check (10.12.26.248) 294.81ms

2023-12-13 15:21:14.072 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 19, in <module>

    client = openai.OpenAI(api_key=api_key)

  File "/home/adminuser/venv/lib/python3.9/site-packages/openai/_client.py", line 92, in __init__

    raise OpenAIError(

openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

2023-12-13 15:21:14.331 503 GET /script-health-check (10.12.26.248) 331.33ms

2023-12-13 15:21:19.030 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 19, in <module>

    client = openai.OpenAI(api_key=api_key)

  File "/home/adminuser/venv/lib/python3.9/site-packages/openai/_client.py", line 92, in __init__

    raise OpenAIError(

openai.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

2023-12-13 15:21:19.243 503 GET /script-health-check (10.12.26.248) 286.88ms

main
bottych8c
ChatGPT
The error message you're encountering indicates that the OpenAI API key is not being set properly in your Streamlit app. This is a common issue when deploying apps that rely on environment variables for configuration, especially for sensitive data like API keys. Here's how to fix it:

Setting the OpenAI API Key in Streamlit Sharing
When deploying a Streamlit app on Streamlit Sharing, you need to set environment variables through their secrets management feature. This ensures that your API key remains secure and is not exposed in your code. Here's what you need to do:

Access the Streamlit Secrets Management:

Go to your Streamlit Sharing dashboard.
Find your app and click on "Edit".
Navigate to the "Secrets" section.
Add Your OpenAI API Key:

In the secrets management area, you need to add your OpenAI API key in a JSON format. For example:
json
Copy code
{
  "OPENAI_API_KEY": "your_actual_openai_api_key_here"
}
Replace your_actual_openai_api_key_here with your actual OpenAI API key.
Modify Your Code to Use the Secret:

In your Streamlit app code, ensure you are accessing the API key from Streamlit's secrets. For example:
python
Copy code
import streamlit as st
# ... other imports ...

# Load the OpenAI API key from Streamlit's secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=api_key)
# ... rest of your code ...
Redeploy Your App:

After setting the secret, redeploy your app for the changes to take effect.
Testing Locally with .env File
If you are testing your app locally and using a .env file for environment variables, ensure that your .env file contains the correct API key and is located in your project's root directory. The file should look something like this:

plaintext
Copy code
OPENAI_API_KEY=your_actual_openai_api_key_here
When deploying, remember not to upload the .env file to public repositories for security reasons.

By following these steps, your Streamlit app should be able to securely access the OpenAI API key and function correctly both locally and on Streamlit Sharing.

User
I got another error:
2023-12-13 15:26:29.102 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 119, in <module>

    main()

  File "/mount/src/gemeinderecht/app.py", line 105, in main

    enhanced_user_query = user_query + " " + relevance_phrases.get(relevance, "")

NameError: name 'relevance_phrases' is not defined

this is the app code:
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
ChatGPT
The error you're encountering is because the relevance_phrases dictionary is referenced in your main() function but it's not defined anywhere in your code. The error message NameError: name 'relevance_phrases' is not defined indicates that Python cannot find this variable.

To resolve this, you need to define the relevance_phrases dictionary in your script. It seems like relevance_phrases should contain mappings similar to relevance_mapping but used in a different context in your main() function.

Here's how you can define it:

python
Copy code
# Define the relevance_phrases dictionary
relevance_phrases = {
    "assembly": "Some description or value for assembly",
    "mail voting": "Some description or value for mail voting",
    "none": "Some description or value for none"
}
Make sure to replace "Some description or value for ..." with the actual descriptions or values you want to associate with each key.

Add this definition at the global level in your script, preferably near the top where you define other global variables, so it's accessible throughout your script, including in the main() function.

Once you've added this definition, the error should be resolved. Ensure that the values in relevance_phrases are consistent with how you want to enhance the user query in the main() function.

User
cant we just use the  relevance_mapping = {
        "assembly": "Die Frage bezieht sich auf Gemeindeversammlungen.",
        "mail voting": "Die Frage bezieht sich auf Wahlen an der Urne.",
        "none": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
    }?
ChatGPT
Certainly! You can use the existing relevance_mapping dictionary in your main() function if it serves the purpose you intended for relevance_phrases. This will make your code more consistent and prevent redundancy. Just make sure the values in relevance_mapping align with how you want to enhance or modify the user's query.

Here's how you can modify your main() function to use relevance_mapping:

python
Copy code
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
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
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
In this updated version, the enhanced_user_query is created by appending a string from relevance_mapping based on the user's selected relevance. This should resolve the NameError you encountered and ensure the app functions as expected.

User
File "/mount/src/gemeinderecht/app.py", line 105, in main

    enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")

NameError: name 'relevance_mapping' is not defined
ChatGPT
The error indicates that relevance_mapping is not defined in the scope of the main function. This can happen if relevance_mapping is defined inside another function or outside the main function's scope. To resolve this issue, ensure that relevance_mapping is defined at the global level in your script, so it is accessible within the main function.

Here's how to adjust your code:

Define relevance_mapping outside of any function, at the top level of your script. This makes it a global variable that can be accessed from any function within the script.

Modify your main function to use this global variable.

Here's how your updated script should look:

python
Copy code
import streamlit as st
# ... other imports ...

# Define relevance_mapping at the global level
relevance_mapping = {
    "assembly": "Die Frage bezieht sich auf Gemeindeversammlungen.",
    "mail voting": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "none": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
}

# ... other code ...

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
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            # ... rest of your code ...

# ... rest of your code ...

if __name__ == "__main__":
    main()
By defining relevance_mapping at the global level, it will be accessible throughout your script, including inside the main function.


2 / 2

User
I get: NameError: name 'lawcontent_dict' is not defined

However, this is my code, I dont see lawcontent_dict anywhere?
import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
relevance_mapping = {
    "assembly": "Die Frage bezieht sich auf Gemeindeversammlungen.",
    "mail voting": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "none": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
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
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

ChatGPT
The error NameError: name 'lawcontent_dict' is not defined seems to stem from a discrepancy in variable naming in your script. The variable lawcontent_dict is mentioned in the error, but in your script, you actually use law_data.

The function generate_prompt and the part of the main function where relevant_lawcontent_dict is assigned both expect a dictionary named lawcontent_dict, but in your script, you have loaded the data into a variable named law_data.

To resolve this, you should align the variable names. You can do this by either:

Renaming law_data to lawcontent_dict wherever it appears in your script.
Updating the generate_prompt function and the main function to use law_data instead of lawcontent_dict.
The second option might be simpler, as it requires fewer changes. Here's how you can modify the generate_prompt function call in main:

python
Copy code
# ... existing code ...

def main():
    # ... existing code in main ...

    if st.button("Generate Prompt"):
        if user_query:
            # ... existing code ...

            # Change 'lawcontent_dict' to 'law_data' here
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)

            st.text_area("Generated Prompt:", prompt, height=300)

    # ... rest of the main function ...

# ... rest of your script ...
And ensure your generate_prompt function is expecting law_data as the argument:

python
Copy code
def generate_prompt(user_query, relevance, top_articles, law_data):
    # ... body of generate_prompt ...
This change should resolve the NameError by ensuring consistency in how you refer to the law data throughout your script.

User
Am I crazy? I already seem to have this changed, here is my code again:
import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
relevance_mapping = {
    "assembly": "Die Frage bezieht sich auf Gemeindeversammlungen.",
    "mail voting": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "none": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
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
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

ChatGPT
Your code indeed seems to be consistent in using law_data across the functions. The generate_prompt function and the main function are both using law_data, and there's no reference to lawcontent_dict.

If you're still encountering the NameError: name 'lawcontent_dict' is not defined error, it could be due to one of the following reasons:

Cached Version: Sometimes, Streamlit or the deployment environment might be running a cached or older version of your script. Try clearing the cache or redeploying the app to ensure the latest version of your code is being used.

Different File or Code Segment: Double-check that the error is indeed coming from this specific script. Sometimes, in a project with multiple files or in a complex deployment environment, an error might be coming from a different part of the code.

External Dependencies or Imported Modules: Ensure that the error is not originating from an external module or a different file that your main script might be importing.

Here are some steps to troubleshoot:

Redeploy: If you're using a platform like Streamlit Sharing, try redeploying your application to ensure it's running the latest version of your code.
Clear Cache: In Streamlit, you can clear the cache by adding st.legacy_caching.clear_cache() at the beginning of your script and running it once.
Check All Files: If your project contains multiple Python files, verify that the error isn't coming from a different file.
Local Testing: Run your Streamlit app locally to see if the error persists. This can help determine if the issue is with the deployment environment.
If the problem continues after these checks, there might be an issue with the deployment environment or a deeper code problem that isn't immediately apparent from the script you've shared.

User
yeeeess, it worked. thanks!!
Now I want to change the app slightly, I want to first display the top matching articels. Also I want change the selection of relevance: The user should not be prompted assembly or mail voting, since its a german app. Can that be changed easyly or is that cumbersome?
ChatGPT
I'm glad to hear that your app is working now! To make the changes you've described, you can certainly adjust the Streamlit app relatively easily. Let's break down the changes:

Display the Top Matching Articles: Before generating the full prompt, you can display the top matching articles based on their similarity scores.

Change the Relevance Selection: You can modify the relevance selection options to better suit your German user base.

Here's how you can implement these changes:

Displaying the Top Matching Articles
After calculating the similarities and sorting the articles, you can display the top articles before generating the prompt. Add a section in your main function to do this:

python
Copy code
# ... existing code ...

def main():
    # ... existing code ...

    if st.button("Generate Prompt"):
        if user_query:
            # ... existing code for processing the query ...

            # Display the top matching articles
            st.subheader("Top Matching Articles")
            for title, score in top_articles:
                st.write(f"{title}: {round(score, 2)}")

            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)
            st.text_area("Generated Prompt:", prompt, height=300)

        else:
            st.warning("Please enter a query.")

# ... rest of your script ...
Changing the Relevance Selection
To modify the relevance selection, you just need to adjust the options in the relevance_options list and how you display them in the selectbox. Here's an example modification with German terms:

python
Copy code
def main():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")

    # German relevance options
    relevance_options = {
        "assembly": "Gemeindeversammlungen",
        "mail voting": "Urnenwahl",
        "none": "Allgemein"
    }
    relevance = st.selectbox("Relevanz auswählen:", list(relevance_options.keys()), format_func=lambda x: relevance_options[x])

    # ... rest of your main function ...
This will display the options in German, and the selection will be passed as the key ('assembly', 'mail voting', 'none') to the rest of your code.

These changes should make your app more user-friendly for your German audience and provide a clearer understanding of the top matching articles before generating the full prompt. Remember to test the app after making these changes to ensure everything works as expected.

User
but remember that we definded a mapping outside the mai-function:
elevance_mapping = {
    "assembly": "Die Frage bezieht sich auf Gemeindeversammlungen.",
    "mail voting": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "none": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."

isnt that a problem?
ChatGPT
You're right to point out the use of relevance_mapping outside the main function. This mapping is used to append a specific text to the user's query based on their relevance selection. If you change the relevance options in the selectbox, you'll need to ensure that the mapping still aligns correctly with these options.

Given that you want to change the relevance options to German, here's what you can do:

Adjust the relevance_options in the selectbox to use German terms.
Make sure that the keys in the relevance_mapping dictionary match these new options.
Update the rest of your code to work with these new keys.
Here's how you can implement these changes:

Updating the Relevance Options and Mapping
First, update your relevance options and the relevance_mapping to match:

python
Copy code
# German relevance options and mapping
relevance_options = {
    "gemeindeversammlungen": "Gemeindeversammlungen",
    "urnenwahl": "Urnenwahl",
    "allgemein": "Allgemein"
}

relevance_mapping = {
    "gemeindeversammlungen": "Die Frage bezieht sich auf Gemeindeversammlungen.",
    "urnenwahl": "Die Frage bezieht sich auf Wahlen an der Urne.",
    "allgemein": "Die Frage ist allgemein und nicht spezifisch relevant für die Gemeindeversammlung oder Urnenwahl."
}
Adjusting the selectbox
Update the selectbox in your main function to use the new options:

python
Copy code
def main():
    # ... existing code ...

    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance = st.selectbox("Relevanz auswählen:", list(relevance_options.keys()), format_func=lambda x: relevance_options[x])

    # ... rest of your main function ...
Ensuring Consistency
Make sure the rest of your code uses these new keys ("gemeindeversammlungen", "urnenwahl", "allgemein") consistently. Specifically, check the logic in the generate_prompt function and anywhere else the relevance_mapping is used to ensure it aligns with these new keys.

By making these adjustments, you maintain the functionality of your app while updating the relevance options to be more appropriate for your German audience. Remember to test the app thoroughly after making these changes to ensure everything works as expected.

User
great, it more or less worked, few changes still:
Regarding this block:
      if relevance == "Gemeindeversammlung":
            applicability = "Dieser § ist direkt auf Gemeindeversammlungen anwendbar." if "Directly Applicable: Assembly" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Gemeindeversammlungen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        elif relevance == "Urnenwahl":
            applicability = "Dieser § ist direkt auf Urnenwahl anwendbar." if "Directly Applicable: Mail Voting" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Urnenwahlen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        else:
            applicability = "Überprüfung der direkten Anwendbarkeit ist nicht erforderlich."

I dont need this senctence in the prompt: Überprüfung der direkten Anwendbarkeit ist nicht erforderlich. Because in the case the user has selected that its not relevant, we dont need to overload the prompt with this info
ChatGPT
To remove the sentence "Überprüfung der direkten Anwendbarkeit ist nicht erforderlich" from the prompt when the user selects the option indicating that the question is not specifically relevant, you can modify the generate_prompt function. Specifically, you can change the else clause in the if-elif-else block that handles the applicability string.

Here's the revised section of your generate_prompt function:

python
Copy code
def generate_prompt(user_query, relevance, top_articles, law_data):
    # ... existing code ...

    for title, _ in top_articles:
        article = law_data.get(title, {})
        # ... existing code ...

        if relevance == "gemeindeversammlungen":
            applicability = "Dieser § ist direkt auf Gemeindeversammlungen anwendbar." if "Directly Applicable: Assembly" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Gemeindeversammlungen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        elif relevance == "urnenwahl":
            applicability = "Dieser § ist direkt auf Urnenwahl anwendbar." if "Directly Applicable: Mail Voting" in article.get("tags", []) else "Dieser § ist nur sinngemäss auf Urnenwahlen anwendbar. Es könnte direkt anwendbare § geben, oder Vorschriften in der Gemeindeordnung zu beachten sein, die nicht bekannt sind."
        else:
            applicability = ""

        prompt += f"\n{article_number}. §: {title} von folgemden Gesetz:{name}\n"
        if applicability:
            prompt += f"   - Anwendbarkeit: {applicability}\n"
        prompt += f"   - **Inhalt:** {content}\n"
        article_number += 1

    # ... rest of the function ...

    return prompt
In this updated version:

The else block sets applicability to an empty string.
The prompt construction only includes the applicability sentence if applicability is not empty.
This change effectively removes the "Überprüfung der direkten Anwendbarkeit ist nicht erforderlich" sentence when it's not relevant.
With these adjustments, your prompt will be more concise and tailored to the user's selection.

User
oke, this is my main function now:
def main():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Gemeindeversammlung", "Urnenwahl", "nicht relevant"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Gemeindeversammlungen oder Urnenwahlen bezieht, oder ob dies nicht relevant ist:", relevance_options)

    # Generate prompt button
    if st.button("Generate Prompt"):
        if user_query:
            # Process the query
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]
            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                st.write(f"{title}: {round(score, 2)}")
            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

Now I want two buttons:
First there should be an "Abschicken" button which after clicking it displays the best matching articles. 
Second there should be the Generate Prompt Button

Also the top matching arictles should not only be displayed by title, but with the content. The content should also remain in the same form as it is in the JSON, meaning there should still be paragraphs and not everything in one line. Can you help me?
ChatGPT
Certainly! To implement the changes you've described, you'll need to adjust your Streamlit app to include two separate buttons and modify the display of the top matching articles. Here's how you can do it:

Two Separate Buttons: "Abschicken" and "Generate Prompt"
"Abschicken" Button: This button will process the query and display the best matching articles.
"Generate Prompt" Button: This button will generate and display the prompt based on the selected articles.
Displaying Article Content with Proper Formatting
For displaying the content of the articles, you'll need to fetch the content from your law_data and ensure it's displayed with the correct formatting (e.g., preserving paragraphs).

Here's how you can modify your main function:

python
Copy code
def main():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Gemeindeversammlung", "Urnenwahl", "nicht relevant"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Gemeindeversammlungen oder Urnenwahlen bezieht, oder ob dies nicht relevant ist:", relevance_options)

    # "Abschicken" button to display top matching articles
    if st.button("Abschicken"):
        if user_query:
            # Process the query for top articles
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                article_content = get_article_content(title, law_data[title]['content'])  # Assuming 'content' holds the paragraphs
                st.write(f"{title} (Score: {round(score, 2)}):\n{article_content}\n\n")
        else:
            st.warning("Please enter a query.")

    # "Generate Prompt" button to create and display the prompt
    if st.button("Generate Prompt"):
        if user_query and top_articles:
            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            st.warning("Please enter a query or find matching articles first.")

if __name__ == "__main__":
    main()
In this updated version:

There are two separate buttons: "Abschicken" for fetching and displaying top articles and "Generate Prompt" for creating the prompt.
The "Generate Prompt" button now checks if top_articles is available (which is populated when "Abschicken" is clicked).
The get_article_content function is assumed to return the content in the desired format, preserving paragraphs. Make sure this function correctly formats the content from your JSON data.
This approach should provide a more user-friendly experience, allowing users to first see the top matching articles before generating a prompt based on them. Make sure to test the changes to ensure the app behaves as expected.

User
ups got an error:
2023-12-14 12:48:51.738 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 132, in <module>

    main()

  File "/mount/src/gemeinderecht/app.py", line 117, in main

    article_content = get_article_content(title, law_data[title]['content'])  # Assuming 'content' holds the paragraphs

KeyError: 'content'

I think we need to change the function:
def get_article_content(title, data):
    # Retrieve the list of paragraphs for the given title, default to an empty list if the title is not found
    paragraphs = data.get(title, [])

    # Combine the title and the paragraphs into a single text string
    return title + '\n' + ' '.join(paragraphs)

First its not data, but law_data, and the nested dict is called inhalt. Also I dont want a single string
ChatGPT
The error you're encountering (KeyError: 'content') indicates that the key 'content' is not present in the law_data[title] dictionary. Based on your update, it seems that the correct key is 'inhalt', and you want to display the paragraphs separately, not as a single string.

Let's update the get_article_content function accordingly:

Change data.get(title, []) to data.get('inhalt', []) to correctly access the 'inhalt' key in your data.
Modify the function to return the paragraphs as a list rather than a single string.
Here's the revised get_article_content function:

python
Copy code
def get_article_content(title, data):
    # Retrieve the list of paragraphs for the given title, default to an empty list if the title is not found
    paragraphs = data.get('inhalt', [])

    # Return paragraphs as a list
    return paragraphs
Now, update the way you display the article content in the main function. Instead of concatenating paragraphs into a single string, you can display them separately:

python
Copy code
def main():
    # ... existing code ...

    if st.button("Abschicken"):
        if user_query:
            # ... code to process the query ...

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                article_content = get_article_content(title, law_data[title])
                st.write(f"{title} (Score: {round(score, 2)}):")
                for paragraph in article_content:
                    st.write(paragraph)
                st.write("")  # Add a space after each article

    # ... rest of the main function ...

if __name__ == "__main__":
    main()
In this updated version:

get_article_content now returns a list of paragraphs.
In the main function, each paragraph is displayed separately, maintaining the formatting as it is in the JSON data.
These changes should resolve the error and display the article content in the desired format. Remember to test the app to ensure it works as expected.

User
 # "Abschicken" button to display top matching articles
    if st.button("Abschicken"):
        if st.button("Abschicken"):
        if user_query:
            # ... code to process the query ...

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                article_content = get_article_content(title, law_data[title])
                st.write(f"{title} (Score: {round(score, 2)}):")
                for paragraph in article_content:
                    st.write(paragraph)
                st.write("")  # Add a space after each article
        if user_query:
            # Process the query for top articles
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                article_content = get_article_content(title, law_data[title]['content'])  # Assuming 'content' holds the paragraphs
                st.write(f"{title} (Score: {round(score, 2)}):\n{article_content}\n\n")
        else:
            st.warning("Please enter a query.")
ChatGPT
It looks like there's a bit of duplication and a misplaced button handler in your code. Let's clean that up so that the "Abschicken" button works as intended. The goal is to have the "Abschicken" button process the user query, calculate the similarities, and display the top matching articles. Here's the revised main function:

python
Copy code
def main():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Gemeindeversammlung", "Urnenwahl", "nicht relevant"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Gemeindeversammlungen oder Urnenwahlen bezieht, oder ob dies nicht relevant ist:", relevance_options)

    # "Abschicken" button to display top matching articles
    if st.button("Abschicken"):
        if user_query:
            # Process the query for top articles
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                article_content = get_article_content(title, law_data[title])  # Fetching content for each article
                st.write(f"{title} (Score: {round(score, 2)}):")
                for paragraph in article_content:
                    st.write(paragraph)
                st.write("")  # Add a space after each article
        else:
            st.warning("Please enter a query.")

    # Rest of your main function, including the "Generate Prompt" button ...

if __name__ == "__main__":
    main()
In this corrected version:

There's only one if st.button("Abschicken"): block, which is the correct way to handle a button click in Streamlit.
The code within this block first checks if user_query is not empty. If it's not, it processes the query, calculates the similarities, and displays the top articles.
The get_article_content function is used to fetch and display the content for each top article, maintaining paragraph formatting.
Make sure to test the app after making these changes to confirm that everything works as expected.

User
3-12-14 12:56:59.012 Uncaught app exception

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script

    exec(code, module.__dict__)

  File "/mount/src/gemeinderecht/app.py", line 132, in <module>

  File "/mount/src/gemeinderecht/app.py", line 117, in main

    for title, score in top_articles:

KeyError: 'content'
ChatGPT
The KeyError: 'content' error suggests that the top_articles list is being used before it's defined or updated, likely due to the structure of your code with respect to the buttons. Additionally, the error message indicates that there's still an attempt to access a 'content' key somewhere in the code, which you mentioned should be 'inhalt'.

To fix these issues, you'll need to:

Ensure top_articles is defined before it's used.
Confirm that you're correctly accessing the 'inhalt' key in the law_data.
Let's adjust your code to address these issues:

python
Copy code
def main():
    st.title("Abfrage des Gesetzes über das Stimm- und Wahlrecht des Kantons Thurgau")

    # User inputs
    user_query = st.text_input("Hier Ihre Frage eingeben:")
    relevance_options = ["Gemeindeversammlung", "Urnenwahl", "nicht relevant"]
    relevance = st.selectbox("Wählen Sie aus, ob sich die Frage auf Gemeindeversammlungen oder Urnenwahlen bezieht, oder ob dies nicht relevant ist:", relevance_options)

    # Initialize an empty list for top_articles
    top_articles = []

    # "Abschicken" button to display top matching articles
    if st.button("Abschicken"):
        if user_query:
            # Process the query for top articles
            enhanced_user_query = user_query + " " + relevance_mapping.get(relevance, "")
            query_vector = get_embeddings(enhanced_user_query)
            relevant_lawcontent_dict = get_relevant_articles(law_data, relevance)
            similarities = calculate_similarities(query_vector, {title: article_embeddings[title] for title in relevant_lawcontent_dict if title in article_embeddings})
            sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_articles = sorted_articles[:5]

            st.subheader("Am besten auf die Anfrage passende Artikel")
            for title, score in top_articles:
                article_content = get_article_content(title, law_data[title])  # Correctly accessing the 'inhalt' key inside get_article_content
                st.write(f"{title} (Score: {round(score, 2)}):")
                for paragraph in article_content:
                    st.write(paragraph)
                st.write("")  # Add a space after each article
        else:
            st.warning("Please enter a query.")

    if st.button("Generate Prompt"):
        if user_query and top_articles:
            # Generate and display the prompt
            prompt = generate_prompt(user_query, relevance, top_articles, law_data)
            st.text_area("Generated Prompt:", prompt, height=300)
        else:
            if not user_query:
                st.warning("Bitte geben Sie eine Anfrage ein.")
            if not top_articles:
                st.warning("Bitte klicken Sie zuerst auf 'Abschicken', um die passenden Artikel zu ermitteln.")
.

if __name__ == "__main__":
    main()




