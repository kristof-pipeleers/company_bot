import streamlit as st
import requests
import math
import os
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.base import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import re

def get_urls(question: str, start_item: int, num: int):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "cx": st.secrets["GOOGLE_SEARCH_ENGINE_ID"],
        "q": f"{question} -filetype:pdf -filetype:xml",
        #"hl": "nl",
        #"gl": "be",
        "num": num,
        #"cr": "Belgium",
        #"lr": "lang_be",
        "key": st.secrets["GOOGLE_API_KEY"],
        "start": start_item
    }
    response = requests.get(url, params=params)
    results = response.json().get('items', [])

    items = []
    for item in results:
        items.append({
            'link': item.get('link', ''),
            'snippet': item.get('snippet', '')
        })

    return items

def google_search_engine(question: str, num_search_results: int):
        if num_search_results > 100:
            raise NotImplementedError('Google Custom Search API supports a max of 100 results')
        elif num_search_results > 10:
            num = 10
            calls_to_make = math.ceil(num_search_results / 10)
        else:
            calls_to_make = 1
            num = num_search_results

        start_item = 1
        items_to_return = []

        while len(items_to_return) < 10:
            items = get_urls(question, start_item, num)
            for item in items:
                print(f"Query: {question}\nURL: {item['link']}\nSnippet: {item['snippet']}")
                # choice = input("Voeg deze URL toe aan de resultaten? (j/n/e): ")
                # if choice.lower() == "j":
                items_to_return.append(item['link'])
                if len(items_to_return) == num_search_results:
                    return items_to_return
                # elif choice.lower() == "e":
                #     return items_to_return
            start_item += 10

        return items_to_return


def crawl_urls(urls):

        def scrape_page(url):
            try:
                # Send a GET request to the URL
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad status codes

                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                content_tags = ['div', 'p', 'article', 'section']
                excluded_classes = ['header', 'footer', 'nav', 'sidebar', 'menu', 'breadcrumb', 'pagination', 'legal', 'advertisement']
                excluded_ids = ['header', 'footer', 'navigation', 'sidebar', 'menu', 'breadcrumbs', 'pagination', 'legal', 'ads']

                unique_texts = set()
                for tag in content_tags:
                    for element in soup.find_all(tag):
                        class_list = element.get('class', [])
                        id_name = element.get('id', '')
                        if not any(excluded in class_list for excluded in excluded_classes) and id_name not in excluded_ids:
                            text_block = re.sub(r'\s+', ' ', element.get_text()).strip()
                            unique_texts.add(text_block)

                text = ' '.join(sorted(unique_texts))

                return text
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                return None
                    
                
        docs = []
        for url in urls:
            content = scrape_page(url)
            if content:
                #docs.append({"url": url, "content": text})
                docs.append(Document(page_content=content, metadata={"url": url}))
            else:
                print(f"Skipped {url}\n")
                
        return docs

def get_result(docs, company):
    
    try:
        client = OpenAI(api_key=os.environ['OPENAI_KEY'], organization=st.secrets["OPENAI_ORG_ID"])
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        splitted_docs = text_splitter.split_documents(docs)

        embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_KEY'])
        
        vectordb = Chroma.from_documents(documents=splitted_docs, embedding=embeddings_model)

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":6})
        
        qa = RetrievalQA.from_chain_type(
            llm=client, 
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True, 
        )

        response_schemas = [
            ResponseSchema(name="raw materials", description="This parameter is true if context is about a mining company, extraction company, mineral extractors, Resource harvesters doing mining, extraction, Extracting, Digging, Quarrying.This represents the mining and extraction non-renewable resources from the different earth layers, such as metal ores, rock, petroleum and natural gas."),
            ResponseSchema(name="refined materials", description="This parameter is true if context is about a refining company, extraction company, Purification firms, Material processors doing refining, Purifying, Processing, Distilling. This represents the refining of raw materials to a high level of purity, such as metals, minerals, different hydrocarbons (gasoline, naphtha, kerosine, …)"),
            ResponseSchema(name="synthetic materials", description="This parameter is true if context is about a material producer, Synthetic manufacturers, Compound fabricators doing material production, Manufacturing, Synthesizing, Compounding. This represents the production of a synthetic material that has altered the chemical state of the molecules that entered the production process, such as all the different kind of art, alloys, …"),
            ResponseSchema(name="part", description="This parameter is true if context is about a converter, Part fabricators, Material converters producing parts, converting materials, Fabricating parts, Transforming materials. This represents the production of a ‘mono material’ object. The object has often no function in standalone mode, such as plastic casing, a spoke, …"),
            ResponseSchema(name="component", description="This parameter is true if context is about a component builder, Assembly specialists, Component constructors producing components, assembling components, Constructing components, Component assembly. This represents the combining of different parts into a functional component, such as a engine, wheel, … A component had a function but it needs to be exerted in the combination with other components."),
            ResponseSchema(name="product manufacturer", description="This parameter is true if context is about a product producer, assembler, Manufacturing firms, Production units assembling, Manufacturing, Constructing. This represents the combining of different components into a functional product, such as a toaster, bike, …"),
            ResponseSchema(name="product distribution - B2B", description="This parameter is true if context is about a wholesaler, distributor, Bulk suppliers, Trade distributors selling, renting, leasing, Distributing, Supplying, Providing. This represents all activities that allow the final user to gain access over the product."),
            ResponseSchema(name="product distribution - B2C", description="This parameter is true if context is about a seller, assembler, Retailers, Merchants selling, renting, leasing, Retailing, Offering. This represents all activities that allow the final user to gain access over the product."),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        query = f"Judge whether the following company activities are true for the company {company}: Raw Material Extraction, Production of Refined Materials, Production of Synthetic Materials, Part, Component, Product Distribution - B2B, Product Distribution - B2C. Antwoord met een json-dict: {format_instructions}"
        result = qa({"query": query})
        print(result['result'].strip())
        result_as_dict = output_parser.parse(result['result'].strip())
        print(f"\nAsnwer for {company}: {result_as_dict}")

        return result_as_dict 
    
    except Exception as e:
        print(f"Error processing results for {company}: {e}")
        return {}


def get_company_strategies(companies):

    strategies = {
        'raw materials': [],
        'refined materials': [],
        'synthetic materials': [],
        'part': [],
        'component': [],
        'product manufacturer': [],
        'product distribution - B2B': [],
        'product distribution - B2C': [],
    }
    
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    total_companies = len(companies)
    for index, company in enumerate(companies):
        # Update the progress bar with the current progress
        current_progress = int((index / total_companies) * 100)
        my_bar.progress(current_progress, text=f"Searching relevant information on the web for :orange[{company}] ...")

        # Your code for processing each company goes here
        urls = google_search_engine(question=company, num_search_results=4)
        docs = crawl_urls(urls)
        result = get_result(docs, company)
        for strategy, is_present in result.items():
            is_present_bool = str(is_present).lower() == 'true'
            if is_present_bool:
                strategies[strategy].append(company)

        # Update the progress bar for the next iteration
        if index == total_companies - 1:
            # Complete the progress bar when processing the last company
            my_bar.progress(100, text="Operation complete.")

    # Optionally, you can clear the progress bar here if you don't want it to stay
    my_bar.empty()
    
    visualization_strs = []
    for category, companies in strategies.items():
        companies_str = ', '.join(companies) if companies else 'None'
        visualization_strs.append(f"<strong>{category}</strong>: {companies_str}<br>")

    visualization = ''.join(visualization_strs)

    return visualization




