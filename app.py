import os
from dotenv import load_dotenv
from function import initialize_db_client, get_db_results, get_model, open_pdf, prompt_generation, send_to_watsonxai, translate_large_text, translate_to_thai, thai_word_fix, question_enrichment_prompt_generation
from flask import Flask, render_template, request
from flask import jsonify
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from googletrans import Translator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


load_dotenv()
MILVUS_HOST = os.environ["MILVUS_HOST"]
MILVUS_PORT = os.environ["MILVUS_PORT"]
MILVUS_USERNAME = os.environ["MILVUS_USERNAME"]
MILVUS_SERVER_NAME = os.environ["MILVUS_SERVER_NAME"]
MILVUS_PASSWORD = os.environ["MILVUS_PASSWORD"]
MILVUS_COLLECTION_ARYA = os.environ["MILVUS_COLLECTION_ARYA"]
MILVUS_COLLECTION_PANG = os.environ["MILVUS_COLLECTION_PANG"]
project_id = os.environ["PROJECT_ID"]
ibm_cloud_url = os.environ["IBM_CLOUD_URL"]
api_key = os.environ["API_KEY"]



creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

llama70b = "meta-llama/llama-2-70b-chat"
llama13b = "meta-llama/llama-2-13b-chat"

# granite13b = "ibm/granite-13b-chat-v1"

enrich_model_params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.MIN_NEW_TOKENS: 3,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.RANDOM_SEED: 42,
    GenParams.TEMPERATURE: 1.0,
    GenParams.REPETITION_PENALTY: 1.0,
    GenParams.STOP_SEQUENCES: ['\n\n']
}    
enrich_model = Model(
    model_id=llama13b,
    params=enrich_model_params,
    credentials=creds,
    project_id=project_id)

model_params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.MIN_NEW_TOKENS: 3,
    GenParams.MAX_NEW_TOKENS: 400,
    GenParams.RANDOM_SEED: 42,
    GenParams.TEMPERATURE: 1.0,
    GenParams.REPETITION_PENALTY: 1.0,
}
llama_model = Model(
    model_id=llama13b,
    params=model_params,
    credentials=creds,
    project_id=project_id)



app = Flask(__name__)
_ = initialize_db_client(MILVUS_HOST, MILVUS_PORT, MILVUS_SERVER_NAME, MILVUS_USERNAME, MILVUS_PASSWORD)
model = get_model(model_name='BAAI/bge-large-en-v1.5', max_seq_length=512)
translator = Translator()

@app.route('/live')
def liveness_check():
    return 'OK', 200

@app.route('/data', methods=['POST'])
def data():
    data = request.json
    print(data)
    clean_data = thai_word_fix(data["query"])
    # clean_data = data["query"]
    print(clean_data)

    translated_question_english = translator.translate(clean_data, dest='en').text



    # enrich_prompt = question_enrichment_prompt_generation(translated_question_english)

    # print(enrich_prompt)


    # response_enrich = send_to_watsonxai(enrich_model,
    #                     prompts=[enrich_prompt],
    #                     model_name=llama13b, 
    #                     max_new_tokens=300,
    #                     min_new_tokens=2
    #                     )
    # print(response_enrich[0])
    enrich_question = translated_question_english
    # enrich_question = response_enrich[0]
    # enrich_question = '\n'.join([response_enrich[0], translated_question_english])
    print("Question for query: \n", enrich_question)
    ## Retrieval
    # query_encode = [list(i) for i in model.encode([enrich_question])]
    # collection = Collection(MILVUS_COLLECTION_ARYA)
    # collection.load()
    # documents = collection.search(data=query_encode, anns_field="embeddings", param={"metric":"IP","offset":0},
    #                 output_fields=["text_to_encode", "law_type", "topic", "detail"], limit=2)
    
    DOCS_ARYA = get_db_results(enrich_question, model, MILVUS_COLLECTION_ARYA, 2)
    DOCS_PANG = get_db_results(enrich_question, model, MILVUS_COLLECTION_PANG, 2)
    print("no. of retrieved docs", len(DOCS_ARYA))

    def formatting_docs(DOCS_ARYA):
        concatenated_docs = ""
        i = 1

        reference_return = ""
        for doc in DOCS_ARYA:
            concatenated_docs += f'Document {i}\n'
            concatenated_docs += f'\n{doc.text_to_encode}\n\n'
            i += 1
            print(doc.distance)
            if doc.distance > 180:
                reference_return += f'ประเภทกฎหมาย: {doc.law_type}\n'.replace('_text', '')
                reference_return += f'ข้อความอ้างอิง: {doc.detail}\n\n'.replace('_text', '')
        return concatenated_docs, reference_return

    concatenated_docs_arya, reference_return_arya = formatting_docs(DOCS_ARYA)
    concatenated_docs_pang, reference_return_pang = formatting_docs(DOCS_PANG)

    # results = {"results": f"Received data modified: {concatenated_docs}"}
    # print(results)

    ## Augmented Generation
    prompt = prompt_generation(concatenated_docs_arya, concatenated_docs_pang, enrich_question)

    response_g = send_to_watsonxai(llama_model,
                            prompts=[prompt],
                            model_name=llama13b,
                            max_new_tokens=300,
                            min_new_tokens=2
                            )

    translate_back = translator.translate(response_g[0], dest='th').text
    results = {"results": translate_back, "results_en": response_g[0], "reference_arya": reference_return_arya, "reference_pang": reference_return_pang}
    # print(results)
    return results

if __name__ == '__main__':
    app.run(debug=True, port=8001, host="0.0.0.0")
