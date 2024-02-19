import requests
import os
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
import streamlit as st
import time
import json
from streamlit.components.v1 import html
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from scrapers import KBO_scraper
import numpy as np

organization_id = st.secrets["OPENAI_ORG_ID"]
assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]
openai_key = st.secrets["OPENAI_KEY"]
google_key = st.secrets["GOOGLE_API_KEY"]

client = OpenAI(api_key=openai_key, organization=organization_id)

file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)
template = env.get_template('system_message.jinja2')

# Render the template with your actual values
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

# Define the HTML and CSS for the custom notification
def custom_notification(status_message):
    if status_message:
        notification_html = f"""
        <style>
        @keyframes spin {{
          0% {{ transform: rotate(0deg); }}
          100% {{ transform: rotate(-360deg); }}
        }}
        .loading-icon {{
          display: inline-block;
          animation: spin 1s linear infinite;
        }}
        .notification-box {{
          padding: 10px;
          margin: 10px 0;
          border-radius: 5px;
          background-color: #f3f4f6;
          color: #111827;
          display: flex;
          align-items: center;
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
def load_openai_client_and_assistant():
    client          = OpenAI(api_key=openai_key)
    my_assistant    = client.beta.assistants.retrieve(assistant_id)
    thread          = client.beta.threads.create()

    return client , my_assistant, thread

client, assistant, assistant_thread = load_openai_client_and_assistant()

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
   
    client = OpenAI(api_key=openai_key, organization=organization_id)
    
    # data = load_data("Nacebel-2008-FR-NL-DE-EN.csv")
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=100,
    #     chunk_overlap=0
    # )
    # splitted_docs = splitter.split_documents(data)

    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)
    
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
def wait_on_run(run, thread, message, status_placeholder):

    while run.status == "queued" or run.status == "in_progress" or run.status == "requires_action":

        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        custom_notification(f"OpenAI status: {run.status}")
        time.sleep(0.5)   
    
        if run.status == "completed":

            # Once the process is complete, clear the notification
            custom_notification("")

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
            # Once the function call is completed, clear the notification
            custom_notification("")
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

    response = wait_on_run(run, assistant_thread, message, status_placeholder)
    return response

if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

def submit():
    st.session_state.user_input = st.session_state.query
    st.session_state.query = ''


# streamlit UI
st.title(":office: Company Search Bot")

st.subheader("Hi! :wave:")
st.text("Ik ben een chatbot die ontworpen is je te helpen bij het vinden van bedrijven in verschillende sectoren en locaties.")
st.text_input("Hoe kan ik je van dienst zijn vandaag?", key='query', on_change=submit)

user_input = st.session_state.user_input

st.write("You entered: ", user_input)

status_placeholder = st.empty()

if user_input:
    result = get_assistant_response(user_input)
    st.header('Assistant', divider='rainbow')
    st.text(result)