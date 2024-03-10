import os
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
import streamlit as st
import json
from companies import get_companies
from company_strategies import get_company_strategies

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)
template = env.get_template('system_message.jinja2')
system_message = template.render()

functions = [
    {
        "name": "get_companies",
        "description": "retrieves the location and industry of the user question",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "This refers to the locaion in which the user intends to find businesses. With the initial letter of the location consistently capitalized."
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
    },
    {
        "name": "get_value_chain",
        "description": "retrieves the position in the value chain of a list of companies",
        "parameters": {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "description": "This is a list of companies retrieved from the user's query.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "companies"
            ]
        }
    }
]   


# initiate assistant ai response
def get_chat_response(db, client, selected_model):
    
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
        functions=functions,
        function_call="auto", 
        temperature=0.9
    )
    response_message = response.choices[0].message

    print(f"ID= {response.id}")

    try:   
        if response_message.function_call:
            function_args = json.loads(response_message.function_call.arguments)
            print(f"\nfunction call arguments: {function_args}")
            function_name = response_message.function_call.name
            # get companies function call
            if function_name == "get_companies":
                response, pydeck_map = get_companies(db, **function_args)
                
                return response, pydeck_map
            # get value chain function call
            elif function_name == "get_value_chain":
                response = get_company_strategies(**function_args)
                return response, None
            
            else: 
                print(f"Function " + function_name + " does not exist")
    except Exception as e:
        print(e)
        response = "⚠️ Er is iets misgegaan bij het openen van de juiste gegevensbronnen voor uw bedrijfszoekopdracht  ⚠️ <br> Zorg ervoor dat u een bestaande bedrijfstak en locatie invoert. 🔍"
        return response, None
    
    return response_message.content, None

def run_app(db, uid):

    # Replicate Credentials
    with st.sidebar:
        st.title("🍃 advAI:green[CE]")
        st.info(" Ik kan je helpen bij het vinden van bedrijven in verschillende sectoren en locaties.🔍", icon="ℹ️")
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

        st.subheader('🤖 Models')
        selected_model = st.sidebar.selectbox('Kies een OpenAI-model', ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'], key='selected_model')
        client = OpenAI(api_key=os.environ['OPENAI_KEY'])

        st.subheader('📖 Informatie')
        st.markdown('Wil je meer informatie over deze chatbot? Neem contact op met info@werecircle.be')

        image_urls = [
            'images/werecircle-logo.png',
            'images/greenaumatic-logo.png'
        ]

        cols = st.columns(len(image_urls))
        for col, url in zip(cols, image_urls):
            col.image(url, width=150)

    # Welcome message
    users_ref = db.collection("users")
    user_doc = users_ref.document(uid).get()
    username = user_doc.to_dict().get("username") if user_doc.exists else None
    if user_doc.exists:
        st.title(f":wave: Welcome, {username}!")

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hoe kan ik u vandaag van dienst zijn?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👤"):
                st.write(message["content"], unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"], avatar="🍃"):
                st.write(message["content"], unsafe_allow_html=True)

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Hoe kan ik u vandaag van dienst zijn?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # User-provided prompt
    if query := st.chat_input(disabled=not api_key):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.write(query, unsafe_allow_html=True)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="🍃"):
            response_placeholder = st.empty()
            map_placeholder = st.empty()
            response, pydeck_map = get_chat_response(db, client=client, selected_model=selected_model)
            response_placeholder.markdown(response, unsafe_allow_html=True)
            if pydeck_map is not None:
                map_placeholder.pydeck_chart(pydeck_map)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

