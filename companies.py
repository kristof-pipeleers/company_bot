import requests
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scrapers import KBO_scraper
import numpy as np
import googlemaps
import pandas as pd
import pydeck as pdk
import datetime
import jwt


def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def get_relevant_NACE(industry, num):
   
    client = OpenAI(api_key=os.environ['OPENAI_KEY'], organization=st.secrets["OPENAI_ORG_ID"])
    
    # data = load_data("Nacebel-2008-FR-NL-DE-EN.csv")
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
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
    # print(f"Relevant Nace codes from retrieval:")
    # for document in result['source_documents']:
    #     print(document.page_content)
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
        "X-Goog-Api-Key": st.secrets["GOOGLE_API_KEY"],
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


def add_to_db(db, industry, location, unique_array):
    # Define collection reference
    company_info_ref = db.collection("company_info")
    users_ref = db.collection("users")  # Assuming users collection exists
    current_params = st.query_params
    token = jwt.decode(current_params.get('session_token'), options={"verify_signature": False})
    user_doc = users_ref.document(token.get("uid")).get()
    user_email = user_doc.to_dict().get("email") if user_doc.exists else None

    for item in unique_array:
        if len(item) == 2:
            company_name, company_address = item

            # Check if the company address already exists in the database
            existing_docs = company_info_ref.where("company_address", "==", company_address.strip()).limit(1).get()

            if len(existing_docs) == 0:
                # If the address does not exist, insert a new record
                company_info_ref.add({
                    "company_name": company_name.strip(),
                    "company_address": company_address.strip(),
                    "queries": [{
                        "industry": industry,
                        "location": location,
                        "users": [user_email],
                        "timestamp": datetime.datetime.now()
                    }]
                })
            else:
                for doc in existing_docs:
                    doc_ref = company_info_ref.document(doc.id)
                    doc_data = doc.to_dict()
                    query_found = False

                    # Iterate through existing queries to find a match
                    for query in doc_data["queries"]:
                        if query["industry"] == industry and query["location"] == location:
                            query_found = True
                            # Add the new user if they're not already in the list
                            if user_email not in query["users"]:
                                query["users"].append(user_email)
                                # Update the timestamp to reflect the latest query
                                query["timestamp"] = datetime.datetime.now()
                            break

                    if not query_found:
                        # If no matching query is found, add a new one
                        doc_data["queries"].append({
                            "industry": industry,
                            "location": location,
                            "users": [user_email],
                            "timestamp": datetime.datetime.now()
                        })

                    # Update the document with the modified or new query data
                    doc_ref.update({"queries": doc_data["queries"]})

    return True


def initialize_pydeck_map(lat_lon_list):
    df = pd.DataFrame(lat_lon_list, columns=['lat', 'lon', 'company_name', 'company_address'])

    layer = pdk.Layer(
        'ScatterplotLayer',
        df,
        get_position=['lon', 'lat'],
        get_color='[200, 30, 0, 160]',
        get_radius=100,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=df['lat'].mean(),
        longitude=df['lon'].mean(),
        zoom=10,
        pitch=0
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v10',
        tooltip={"html": "<b>Company Name:</b> {company_name}<br><b>Address:</b> {company_address}", "style": {"color": "white"}}
    )
    return r


def generate_pydeck_map(unique_array):
    map_client = googlemaps.Client(key=st.secrets["GOOGLE_API_KEY"])

    lat_lon_list = []
    for item in unique_array:
        if len(item) == 2:
            company_name, company_address = item

            response = map_client.geocode(company_address)
            if response:
                location = response[0]['geometry']['location']
                latitude = location['lat']
                longitude = location['lng']
                lat_lon_list.append([latitude, longitude, company_name, company_address])
    
    pydeck_map = initialize_pydeck_map(lat_lon_list)
    return pydeck_map


# Set openAi client , assistant ai and assistant ai thread
def get_companies(db, location, area, industry):
    
    with st.spinner(f"Retrieving relevant NACE codes for industry: {industry} ..."):
        nace_dict = get_relevant_NACE(industry, 10)
        nace_codes = nace_dict['nace_codes']

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
    
    with st.spinner(f"Inserting company data into database..."):
        if add_to_db(db, industry, location, unique_array):
            print(f"company information for {industry} in {location} is inserted")
        else:
            print(f"Error while interting company information for {industry} in {location}. Is the SQL database running?")
    
    with st.spinner(f"Generating Map..."):
        pydeck_map = generate_pydeck_map(unique_array)
    
    result += '<br>'.join([f"{item[0]} - {item[1]}" for item in unique_array])

    return result, pydeck_map
    
