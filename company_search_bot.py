import requests
import os
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
import streamlit as st
import time
import json
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from scrapers import KBO_scraper
import numpy as np
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

organization_id = st.secrets["OPENAI_ORG_ID"]
google_key = st.secrets["GOOGLE_API_KEY"]

file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)
template = env.get_template('system_message.jinja2')
system_message = template.render()

function_get_companies = {
        "name": "get_companies",
        "description": "retrieves the location and industry of the user question",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "This is the location name or ZIP code where the user wishes to search for businesses."
                },
                "industry": {
                    "type": "string",
                    "description": "This refers to the sector in which the user intends to find businesses."
                }
            },
            "required": [
                "location",
                "industry"
            ]
        }
    }

def custom_notification(status_message):
    if status_message:
        notification_html = f"""
        <style>
        @keyframes spin {{
          0% {{ transform: rotate(360deg); }}
          100% {{ transform: rotate(0deg); }}
        }}
        .loading-icon {{
          display: inline-block;
          animation: spin 1s linear infinite;
        }}
        .notification-box {{
          padding: 0px;
          margin: 0px 20px 0px 10px;
          color: #111827;
          display: flex;
          align-items: top;
        }}
        .notification-text {{
          margin-left: 10px;
        }}
        </style>

        <div class="notification-box">
          <div class="loading-icon">🔄</div>
          <div class="notification-text">{status_message}</div>
        </div>
        """
        status_placeholder.markdown(notification_html, unsafe_allow_html=True)
    else:  # If the status message is empty, clear the notification
        status_placeholder.empty()

# Set openAi client , assistant ai and assistant ai thread
@st.cache_resource
def load_openai_client_and_assistant(llm):
    
    client          = OpenAI(api_key=os.environ['OPENAI_KEY'])
    my_assistant = client.beta.assistants.create(
            instructions=system_message,
            model=llm,
            tools=[{"type": "function", "function": function_get_companies}],
            name=f"Company Search Bot"
        )
    thread          = client.beta.threads.create()

    return client, my_assistant, thread

client = None
assistant = None
assistant_thread = None

def get_companies(location, industry):
    
    kbo_data_array = get_KBO_companies(location, industry)
    print(f'kbo: {kbo_data_array}')
    maps_data_array = get_google_maps_companies(location, industry)
    print(f'maps: {maps_data_array}')

   # Convert lists to numpy arrays if they are not already
    kbo_data_array = np.array(kbo_data_array) if isinstance(kbo_data_array, list) else kbo_data_array
    maps_data_array = np.array(maps_data_array) if isinstance(maps_data_array, list) else maps_data_array

    # Check if either array is empty
    if kbo_data_array.size == 0:
        company_data = maps_data_array
    elif maps_data_array.size == 0:
        company_data = kbo_data_array
    else:
        # Both arrays have data, concatenate them
        company_data = np.concatenate((kbo_data_array, maps_data_array), axis=0)

    # Proceed with finding unique companies if company_data is not empty
    if company_data.size > 0:
        _, unique_indices = np.unique(company_data[:, 1], return_index=True)
        unique_array = company_data[unique_indices]
    else:
        # Return an empty numpy array if both input arrays were empty
        unique_array = np.array([])
    result = '\n'.join([f"{item[0]} - {item[1]}" for item in unique_array])
    return result  
    
def get_KBO_companies(location, industry):
    
    nace_dict = get_relevant_NACE(industry, 5)
    nace_codes = nace_dict['nace_codes']
    custom_notification(f"Relevant NACE codes found: {nace_codes}")

    custom_notification(f"Retrieving company information from KBO database ... (this may take a few minutes)")
    kbo_data_array = KBO_scraper.main([location], nace_codes)
    return kbo_data_array
    
@st.cache_data
def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def get_relevant_NACE(industry, num):

    from langchain_openai import OpenAI

    custom_notification(f"Retrieving relevant NACE codes for your question ...")
   
    client = OpenAI(api_key=os.environ['OPENAI_KEY'], organization=organization_id)
    
    # data = load_data("Nacebel-2008-FR-NL-DE-EN.csv")
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=100,
    #     chunk_overlap=0
    # )
    # splitted_docs = splitter.split_documents(data)

    embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_KEY'])
    
    persist_dir = "NACE_embedding"
    # if not os.path.exists(persist_dir):
    #     os.makedirs(persist_dir)
    # vectordb = Chroma.from_documents(documents=splitted_docs, embedding=embeddings_model, persist_directory=persist_dir)
    # vectordb.persist()
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings_model)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":num})
    
    qa = RetrievalQA.from_chain_type(
        llm=client, 
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, 
    )
    
    nace_code_schema = ResponseSchema(name="nace_codes", description=f"An array of the {num} nace codes")
    description_schema = ResponseSchema(name="descriptions", description="An array of the corresponding descriptions of the nace codes")
    output_parser = StructuredOutputParser.from_response_schemas([nace_code_schema, description_schema])
    format_instructions = output_parser.get_format_instructions()
    
    query = f"Geef de {num} meest relevante NACE codes voor de {industry} sector? Antwoord met een json-dict: {format_instructions}"

    result = qa({"query": query})
    print(result['result'].strip())
    result_as_dict = output_parser.parse(result['result'].strip())
    return result_as_dict


def get_google_maps_companies(location, industry):
    
    custom_notification(f"Retrieving relevant companies from Google Maps...")
    
    # Define the base URL for the Text Search request
    base_url = 'https://places.googleapis.com/v1/places:searchText'
    
    # Construct the search query combining the location and industry
    text_query = f"{industry} in {location}"
    print(text_query)
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": f"{google_key}",
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.priceLevel",
    }

    data = {
        "textQuery": f"{text_query}"
    }

    response = requests.post(base_url, json=data, headers=headers)

    if response.status_code == 200:
        print("Request was successful!")
        company_data_list = []
        for place in response.json()["places"]:
            place_name = place['displayName']['text']
            place_addr = place['formattedAddress']
            company_data_list.append([place_name, place_addr])

        company_data_array = np.array(company_data_list)
        return company_data_array
    else:
        print("Request failed with status code:", response.status_code)
        return response.text

# check in loop  if assistant ai parse our request
def wait_on_run(run, thread, message):

    while run.status == "queued" or run.status == "in_progress" or run.status == "requires_action":

        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        custom_notification(f"OpenAI status: {run.status}")
        time.sleep(0.5)   
    
        if run.status == "completed":

            # Retrieve all the messages added after our last user message
            messages = client.beta.threads.messages.list(
                thread_id=assistant_thread.id, order="asc", after=message.id
            )

            response = messages.data[0].content[0].text.value
            return response

        elif run.status == "requires_action":
            required_actions = run.required_action.submit_tool_outputs.model_dump()
            tool_outputs = []
            for action in required_actions["tool_calls"]:
                func_name = action["function"]["name"]
                arguments = json.loads(action["function"]["arguments"])
                if func_name == "get_companies":
                    response = get_companies(arguments["location"], arguments["industry"])
                    tool_outputs.append({
                        "tool_call_id": action["id"],
                        "output": response
                    })
                else:
                    print("function not found")

            print("Submitting outputs back to the Assistant...")
            
            # Send the function call response back to the LLM?
            # client.beta.threads.runs.submit_tool_outputs(
            #     thread_id=thread.id,
            #     run_id=run.id,
            #     tool_outputs=tool_outputs
            # )           

            client.beta.threads.runs.cancel(
                run_id=run.id, 
                thread_id=thread.id
            )
            
            custom_notification(f"OpenAI Status: {run.status}")

            return response
        
        else:
            print("Waiting for the Assistant to process...")
            time.sleep(2)
    

# initiate assistant ai response
def get_assistant_response(user_input=""):

    message = client.beta.threads.messages.create(
        thread_id=assistant_thread.id,
        role="user",
        content=user_input,
    )
    run = client.beta.threads.runs.create(
        thread_id=assistant_thread.id,
        assistant_id=assistant.id,
    )
    
    response = wait_on_run(run, assistant_thread, message)
    return response
    

# Replicate Credentials
with st.sidebar:
    st.title(":office: Company Search Bot")
    st.subheader("Ik kan je helpen bij het vinden van bedrijven in verschillende sectoren en locaties.")
    if 'OPENAI_KEY' in st.secrets:
        st.success('API-sleutel al verstrekt!', icon='✅')
        api_key = st.secrets['OPENAI_KEY']
    else:
        api_key = st.text_input('Voer je OpenAI API-token in:', type='password')
        if not (api_key.startswith('sk-') and len(api_key)==51):
            st.warning('Voer je OpenAI API-token in!', icon='⚠️')
        else:
            st.success('Ga verder met het invoeren van je prompt!', icon='👉')
    os.environ['OPENAI_KEY'] = api_key

    st.subheader('Models en informatie')
    selected_model = st.sidebar.selectbox('Kies een OpenAI-model', ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'], key='selected_model')
    client, assistant, assistant_thread = load_openai_client_and_assistant(selected_model)
    
    st.markdown('📖 Wil je meer informatie over deze chatbot? Neem contact op met info@werecircle.be')

    image_urls = [
        'images/werecircle-logo.png',
        'images/greenaumatic-logo.png'
    ]

    cols = st.sidebar.columns(len(image_urls))
    for col, url in zip(cols, image_urls):
        col.image(url, width=150)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! :wave: Hoe kan ik u vandaag van dienst zijn?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"], avatar="🍃"):
            st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! :wave: Hoe kan ik u vandaag van dienst zijn?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if query := st.chat_input(disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.write(query)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🍃"):
        status_placeholder = st.empty()
        response = get_assistant_response(user_input=query)
        custom_notification("")
        placeholder = st.empty()
        full_response = ''
        for item in response:
            full_response += item
            placeholder.markdown(full_response)
        placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

