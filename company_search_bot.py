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
from google.cloud.sql.connector import Connector
import sqlalchemy
import pymysql
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

organization_id = st.secrets["OPENAI_ORG_ID"]
google_key = st.secrets["GOOGLE_API_KEY"]
db_password = st.secrets["DB_PASSWORD"]

file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)
template = env.get_template('system_message.jinja2')
system_message = template.render()

function_get_companies = [
    {
        "name": "get_companies",
        "description": "retrieves the location and industry of the user question",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1,
                        "maxItems": 5,
                        "description": "If the user specifies a broader region than a city, lists the 5 largest cities in that region",
                },
                "area": {
                    "type": "boolean",
                    "description": "This refers to the size of the location area. If the user literally states to search in a city without the adjacent areas, this parameter is false. If the user also explicitly asks for the city and the surrounding area, this parameter is true."
                },
                "industry": {
                    "type": "string",
                    "description": "This refers to the sector in which the user intends to find businesses. This should be a detailed description of the sector and not a single word."
                }
            },
            "required": [
                "location",
                "area",
                "industry"
            ]
        }
    }
]  
    
def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def get_relevant_NACE(user_input, industry, num):

    from langchain_openai import OpenAI
   
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
    
    query = f"Geef de 5 meest relevante NACE codes voor de {industry} sector op basis van de meegegeven informatie? Antwoord met een json-dict: {format_instructions}"

    result = qa({"query": query})

    # Print the 'page_content' of each document
    print(f"Relevant Nace codes from retrieval:")
    for document in result['source_documents']:
        print(document.page_content)
    result_as_dict = output_parser.parse(result['result'].strip())
    print(f"\nRelevante NACE codes en beschrijving voor {industry}: {result_as_dict}")
    return result_as_dict

def get_google_maps_companies(location, industry):
    
    # Define the base URL for the Text Search request
    base_url = 'https://places.googleapis.com/v1/places:searchText'
    
    # Construct the search query combining the location and industry
    text_query = f"{industry} in {location}"
    
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
        print("Google API request was successful!")
        company_data_list = []
        for place in response.json()["places"]:
            place_name = place['displayName']['text']
            place_addr = place['formattedAddress']
            company_data_list.append([place_name, place_addr])

        company_data_array = np.array(company_data_list)
        return company_data_array
    else:
        print("Google API request failed with status code:", response.status_code)
        return response.text    

# Set openAi client , assistant ai and assistant ai thread
def get_companies(user_input, location, area, industry):
    
    with st.spinner(f"Retrieving relevant NACE codes for industry: {industry} ..."):
        nace_dict = get_relevant_NACE(user_input, industry, 10)
        nace_codes = nace_dict['nace_codes']

    with st.spinner(f"Retrieving company information from KBO database ... (this may take a few minutes)"):
        kbo_data_array = KBO_scraper.main([location], area, nace_codes)
        print(f'kbo: {kbo_data_array}')
    
    with st.spinner(f"Retrieving relevant companies from Google Maps..."):
        maps_data_array = get_google_maps_companies(location, industry)
        print(f'maps: {maps_data_array}')

   # Convert lists to numpy arrays if they are not already
    kbo_data_array = np.array(kbo_data_array) if isinstance(kbo_data_array, list) else kbo_data_array
    maps_data_array = np.array(maps_data_array) if isinstance(maps_data_array, list) else maps_data_array

    result = ''
    # Check if either array is empty
    if kbo_data_array.size == 0:
        company_data = maps_data_array
        result = '⚠️ Geen resultaten gevonden in de KBO databank ⚠️ <br> Zorg ervoor dat u een duidelijke omschrijving geeft van de gewenste branche en locatie. 🔍 <br><br>'
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
    
    add_to_db(unique_array)
    print(f"company information for {industry} in {location} is inserted")
    
    result += '<br>'.join([f"{item[0]} - {item[1]}" for item in unique_array])

    return result

def add_to_db(unique_array):

    connector = Connector()

    def get_conn():
        conn = connector.connect(
            "socs-414314:us-central1:socs-sql",
            "pymysql",
            user="root",
            password=db_password,
            db="socs-db"
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=get_conn,
    )

    insert_stmt = sqlalchemy.text(
        "INSERT INTO company_info (id, company_name, company_address) VALUES (:id, :company_name, :company_address)",
    )

    select_stmt = sqlalchemy.text(
        "SELECT COUNT(*) FROM company_info WHERE company_address = :company_address"
    )

    with pool.connect() as db_conn:
        db_conn.execute(sqlalchemy.text("CREATE TABLE IF NOT EXISTS company_info (id INT AUTO_INCREMENT PRIMARY KEY, company_name VARCHAR(200), company_address VARCHAR(200))"))

        for item in unique_array:
            if len(item) == 2:
                company_name, company_address = item

                # Check if the address already exists
                existing_count = db_conn.execute(select_stmt, {"company_address": company_address.strip()}).scalar()
                
                # If the address does not exist, insert the new record
                if existing_count == 0:
                    db_conn.execute(
                        insert_stmt,
                        {"id": None, "company_name": company_name.strip(), "company_address": company_address.strip()}
                    )
                else:
                    print(f"Address {company_address} already exists in the database.")

        # Optionally, fetch and print all data for verification
        db_conn.commit()
        data = db_conn.execute(sqlalchemy.text("SELECT * FROM company_info;")).fetchall()
        print(data)


# initiate assistant ai response
def get_chat_response(user_input=""):
    
    dialogue = []
    dialogue.append({"role": "system", "content": system_message})
    for dict_message in st.session_state.messages:
        dialogue.append({
            "role": dict_message["role"], 
            "content": dict_message["content"]
        })
    
    response = client.chat.completions.create(
        model=selected_model,
        messages=dialogue,
        functions=function_get_companies,
        function_call="auto", 
        temperature=0.9
    )
    response_message = response.choices[0].message

    try:   
        if response_message.function_call:
            function_name = response_message.function_call.name
            if function_name == "get_companies":
                function_args = json.loads(response_message.function_call.arguments)
                print(f"\n function call arguments: {function_args}")
                response = get_companies(user_input, **function_args)
                
                return response
            else: 
                print(f"Function " + function_name + " does not exist")
    except Exception as e:
        print(e)
        response = "⚠️ Er is iets misgegaan bij het openen van de juiste gegevensbronnen voor uw bedrijfszoekopdracht  ⚠️ <br> Zorg ervoor dat u een bestaande bedrijfstak en locatie invoert. 🔍"
        return response
    
    return response_message.content

# Replicate Credentials
with st.sidebar:
    st.title(":office: Company Search Bot")
    st.info("Ik kan je helpen bij het vinden van bedrijven in verschillende sectoren en locaties.", icon="ℹ️")
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
    client = OpenAI(api_key=os.environ['OPENAI_KEY'])

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
            st.write(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"], avatar="🍃"):
            st.write(message["content"], unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! :wave: Hoe kan ik u vandaag van dienst zijn?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if query := st.chat_input(disabled=not api_key):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.write(query, unsafe_allow_html=True)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🍃"):
        status_placeholder = st.empty()
        response = get_chat_response(user_input=query)
        status_placeholder.markdown(response, unsafe_allow_html=True)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

