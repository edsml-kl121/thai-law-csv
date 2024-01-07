from typing import Literal, Optional, Any
import numpy as np
import pandas as pd
import glob
import pandas as pd
import os
import ast
import json
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from dotenv import load_dotenv
import os
from googletrans import Translator
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings,
                                  HuggingFaceEmbeddings)
from langchain.vectorstores import FAISS, Chroma, Milvus
from langchain.embeddings import SentenceTransformerEmbeddings

from pymilvus import connections,utility,Collection,CollectionSchema, FieldSchema,DataType
from sentence_transformers import SentenceTransformer, models
import fitz
from PIL import Image
from googletrans import Translator
import json
import requests
import textwrap
from dotenv import load_dotenv
import os
from transformers import AutoProcessor, SeamlessM4TModel
import time
import joblib


load_dotenv()
MILVUS_HOST = os.environ["MILVUS_HOST"]
MILVUS_PORT = os.environ["MILVUS_PORT"]
MILVUS_USERNAME = os.environ["MILVUS_USERNAME"]
MILVUS_SERVER_NAME = os.environ["MILVUS_SERVER_NAME"]
MILVUS_PASSWORD = os.environ["MILVUS_PASSWORD"]
# MILVUS_COLLECTION = os.environ["MILVUS_COLLECTION_PANG"]
project_id = os.environ["PROJECT_ID"]
ibm_cloud_url = os.environ["IBM_CLOUD_URL"]
api_key = os.environ["API_KEY"]
cert_path = os.environ["MILVUS_CERT_PATH"]
def initialize_db_client(MILVUS_HOST, MILVUS_PORT, MILVUS_SERVER_NAME, MILVUS_USERNAME, MILVUS_PASSWORD):
    """
    Initializes and returns a chromadb client.

    Parameters:
    - host (str): The host for the chromadb service. Default is 'localhost'.
    - port (int): The port for the chromadb service. Default is 8000.

    Returns:
    - chromadb.HttpClient: An initialized chromadb client.
    """
    return connections.connect("default", host=MILVUS_HOST,
                    port=MILVUS_PORT, secure=True, server_pem_path=cert_path,
                    server_name=MILVUS_SERVER_NAME,user=MILVUS_USERNAME, password=MILVUS_PASSWORD)

    # return connections.connect(host=host,port=port)


def get_db_results(query_text, model, collection_name="promotion_collection_scb", n_results=4):
    """
    Queries the given collection in the database with the provided query text and returns results.

    Parameters:
    - query_text (str): The text to be queried.
    - collection_name (str): The name of the collection to query. Default is "law_topics".
    - n_results (int): Number of results to fetch. Default is 1.

    Returns:
    - dict: Query results.
    """

    # client = initialize_db_client()
    query_encode = [list(i) for i in model.encode([query_text])]
    collection = Collection(collection_name)
    collection.load()
    documents = collection.search(data=query_encode, anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text_to_encode", "law_type", "topic", "detail"], limit=n_results)
    return documents[0]

def get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768, condition=True):
    if condition:
        # model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        # model_name = "hkunlp/instructor-large"
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode='cls') # We use a [CLS] token as representation
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def open_pdf(pdf_path, page_num):
    # Opening the PDF file and creating a handle for it
    file_handle = fitz.open(pdf_path)

    # The page no. denoted by the index would be loaded
    page = file_handle[page_num]

    # Set the desired DPI (e.g., 200)
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

    # Obtaining the pixelmap of the page
    page_img = page.get_pixmap(matrix=mat)

    # Saving the pixelmap into a png image file
    page_img.save('PDF_page_high_res.png')

    # Reading the PNG image file using pillow
    img = Image.open('PDF_page_high_res.png')

    # Displaying the png image file using an image viewer
    img.show()

def question_enrichment_prompt_generation(question):
    prompt = f"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest PARAPHASE BOT for HR department.
You will recieve INPUT from user in the ''' below.
'''
INPUT: {question}
'''    
PARAPHASE the INPUT using the synonyms of words, remain the same meaning of an INPUT but use different words as much as possible.
<</SYS>>
{question}
PARAPHASE:
[/INST]  
    """    
    return prompt

def prompt_generation(arya, pang, question):
    prompt = f"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest QA Law advisor system.
You will recieve criminal law document CRIMINAL LAW, civil law document CIVIL LAW, and QUESTION from user in the ''' below.
'''
CRIMINAL LAW: {arya},
CIVIL LAW: {pang},
QUESTION: {question}
'''
Answer the QUESTION use criminal law from CRIMINAL LAW and cival law from CIVIL LAW if the question is not related to REFERECE please Answer
"I don't know the answer, because the criminal law is not exist"

AVOID the new line as much as possible.
Answer in brief and concise.
<</SYS>>
QUESTION: {question}
[/INST]
    """
    return prompt

def send_to_watsonxai(model,
                    prompts,
                    model_name="google/flan-ul2",
                    decoding_method="greedy",
                    max_new_tokens=30,
                    min_new_tokens=2,
                    temperature=1.0,
                    repetition_penalty=1.0
                    ):
    '''
   helper function for sending prompts and params to Watsonx.ai

    Args:
        prompts:list list of text prompts
        decoding:str Watsonx.ai parameter "sample" or "greedy"
        max_new_tok:int Watsonx.ai parameter for max new tokens/response returned
        temperature:float Watsonx.ai parameter for temperature (range 0>2)
        repetition_penalty:float Watsonx.ai parameter for repetition penalty (range 1.0 to 2.0)

    Returns: None
        prints response
    '''

    assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"

    # Instantiate parameters for text generation



    # Instantiate a model proxy object to send your requests
    output = []
    for prompt in prompts:
        o = model.generate_text(prompt)
        output.append(o)
    return output




load_dotenv()
neural_seek_url = os.getenv("NEURAL_SEEK_URL", None)
neural_seek_api_key = os.getenv("NEURAL_SEEK_API_KEY", None)

def thai_word_fix(sentence):
    clean_data = sentence.replace('กิจ','กิจธุระ')
    clean_data = clean_data.replace('ฌาปนกิจ','เผาศพ งานศพ')
    clean_data = clean_data.replace('บริษัทสามารถเลิกจ้าง','สามารถเลิกจ้าง')
    clean_data = clean_data.replace('พนักงานสำนักงาน','พนักงานสำนักงาน (หมายถึง วันจันทร์ถึงวันศุกร์)')
    clean_data = clean_data.replace('พนักงานกะ','พนักงานกะ (หมายถึง ทำงาน 5 วันต่อสัปดาห์)')
    return clean_data

# def translate_to_thai(sentence, choice):
#     url = neural_seek_url  # Replace with your actual URL
#     headers = {
#         "accept": "application/json",
#         "apikey": neural_seek_api_key,  # Replace with your actual API key
#         "Content-Type": "application/json"
#     }
#     if choice == True:
#         target = "th"
#     else:
#         target = "en"
#     data = {
#         "text": [
#             sentence
#         ],
#         "target": target
#     }
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     try:
#         output = json.loads(response.text)['translations'][0]
#     except:
#         print(data)
#         if target == "th":
#             output = translate_to_thai_facebook(sentence, True)
#         elif target == "en":
#             output = translate_to_thai_facebook(sentence, False)
#         print("facebook translator: ", output)
#         # print(response)
#     return output

def translate_to_thai(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Thai based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Thai. If False, translates to English.
    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Thai
            translate = translator.translate(sentence, dest='th')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e




def cache_model_components(processor_path='cache/processor.joblib', model_path='cache/model.joblib'):
    """
    This function checks for the cached processor and model. If they are not found, it loads them from the pretrained source and caches them.
    Parameters:
    - processor_path: The path to save or load the processor cache.
    - model_path: The path to save or load the model cache.
    
    Returns:
    - processor: The loaded or cached processor.
    - model: The loaded or cached model.
    - initialization_time: The time taken to initialize or load the components.
    """
    start = time.time()

    # Load processor from cache if available, otherwise from pretrained
    if os.path.exists(processor_path):
        processor = joblib.load(processor_path)
    else:
        processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
        joblib.dump(processor, processor_path)

    # Load model from cache if available, otherwise from pretrained
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")
        joblib.dump(model, model_path)

    end = time.time()
    initialization_time = end - start

    return processor, model, initialization_time




def translate_to_thai(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Thai based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Thai. If False, translates to English.
    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Thai
            translate = translator.translate(sentence, dest='th')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e
        

def translate_large_text(text, translate_function, choice, max_length=500):
    """
    Break down large text, translate each part, and merge the results.
    :param text: str, The large body of text to translate.
    :param translate_function: function, The translation function to use.
    :param max_length: int, The maximum character length each split of text should have.
    :return: str, The translated text.
    """

    # Split the text into parts of maximum allowed character length.
    text_parts = textwrap.wrap(text, max_length, break_long_words=True,
                               replace_whitespace=False)

    translated_text_parts = []

    for part in text_parts:
        # Translate each part of the text.
        translated_part = translate_function(part,
                                             choice)  # Assuming 'False' is a necessary argument in the actual function.
        translated_text_parts.append(translated_part)

    # Combine the translated parts.
    full_translated_text = ' '.join(translated_text_parts)

    return full_translated_text
