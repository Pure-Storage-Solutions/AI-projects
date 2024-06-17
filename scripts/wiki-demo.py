#!/usr/bin/python3

print("Loading libraries and models ...")
from pymilvus import connections, Collection
from transformers import TextStreamer
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import LlamaForCausalLM, LlamaTokenizer
import gradio as gr
import time
import os
import base64

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("HF Token Loaded properly")

else:
    print("Could not find HF Token!")
    exit()
collection_name = "wikipedia"

# Connect to milvus DB
#connections.connect()
connections.connect(host="10.233.47.175", port=19530)
collection = Collection(collection_name)
collection.load()

def get_milvus_context(query: str) -> str:
    '''
    Function to grab relevant vectors from database that match given query
    input: user query (str)
    output: most relevant passages to query (str)
    '''

    # Get embedding of query
    query_embedding = embedding_model.encode(query)

    # Define search params
    search_params = {
    "metric_type": "L2",
    "offset": 0,
    "ignore_growing": False,
    "params": {"nprobe": 10}
    }

    # Grab results of query
    results = collection.search(
        data = [query_embedding],
        anns_field = 'embeddings',
        param = search_params,
        limit = 5,
        expr = None,
        output_fields = ['text']
    )

    # iterate through results to get one large context for query
    hits = results[0]
    context = ""

    for hit in hits:
        context += f"{hit.entity.get('text')}\n\n"

    return context


# Load the chat model
chat_model_name = "Llama-2-13b-chat-hf"

chat_tokenizer = LlamaTokenizer.from_pretrained(chat_model_name, device_map='auto', token=hf_token)
chat_model = LlamaForCausalLM.from_pretrained(chat_model_name, device_map='auto', token=hf_token)
chat_model.eval()
streamer = TextStreamer(chat_tokenizer, skip_prompt=True)

# Load embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(embedding_model_name, device='cuda')

def get_llama_prompt_template(context, query):
    template = f'''<s>[INST] <<SYS>>
You are a helpful assistant. Your tasks is to summarize information related to the user's query.

If context is provided, carefully analyze the data, information, and any relevant facts. Summarize your findings succinctly, focusing on key details provided in the context and stating the most important facts related to the query. Ensure your analysis is directly addressing the user's query with insightful observations and evidence-based conclusions. If no context is provided state that you cannot answer the user's query.
<</SYS>>
<s>[INST]
Context: {context}
Query: {query}
Based on the context provide a well-reasoned and succinct answer that incorporates critical facts and insights. Only the most relevant and helpful answer should be returned, without any extraneous information.
[/INST]
'''
    return template

def summarize_embeddings_template(context_docs):
    template = f'''<s>[INST] <<SYS>>
As a helpful assistant, your task is to summarize information about a multitude of topics. Summarize your findings succinctly, focusing on key facts, historical events, and any important passages that are related to the user's query. Ensure your summary is directly quoting the given context and repeating any key facts. <</SYS>>
Based on the context, provide a well-reasoned and succinct answer that summarizes critical facts and insights. Only the most relevant and helpful answer should be returned, without any extraneous information.
[/INST]
'''
    return template

def generate_response(user_query, include_context):

    if include_context:

        # time vector retrieval
        begin_t = time.time()
        context_docs = get_milvus_context(user_query)
        q_t = time.time() - begin_t
        print(f"Time spent vector query: {q_t:.2f}\n")

        max_tokens = 1024 # more tokens for richer output

        # Update the prompt with the new context docs for summarization
        context_prompt = summarize_embeddings_template(context_docs)

        # Tokenize inputs and pass to GPU
        inputs = chat_tokenizer(context_prompt, return_tensors='pt')
        inputs = inputs.to('cuda')

        # Generate chat response for doc summarization
        res = chat_model.generate(**inputs, streamer=streamer, max_new_tokens=max_tokens)
        response = chat_tokenizer.decode(res[0], skip_special_tokens=True)
        output = response.split("[/INST]")
        context = output[1].strip()

    else:
        context = "NO CONTEXT PROVIDED"
        max_tokens = 128 # less tokens for quicker output


    # Update the prompt with the new context and query
    updated_prompt = get_llama_prompt_template(context, user_query)

    # Tokenize inputs and pass to GPU
    inputs = chat_tokenizer(updated_prompt, return_tensors='pt')
    inputs = inputs.to('cuda')

    # Generate chat response
    res = chat_model.generate(**inputs, streamer=streamer, max_new_tokens=max_tokens)
    response = chat_tokenizer.decode(res[0], skip_special_tokens=True)
    output = response.split("[/INST]")
    return output[1].strip()

def image_to_base64(image_path):
    # make image readable by gradio
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


image_path = 'ps-logo-sec-digital-om-lt.png'
base64_string = image_to_base64(image_path)

title_with_image = f"<img src='data:image/png;base64,{base64_string}' width='400' height='300'>"

description_with_style = f'''
<div style="font-size: 20px;">
Financial Analysis Virtual Assistant
</div>
'''

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Enter your question"),
        gr.Checkbox(label="Connect SEC Filings to AI Engine", value=False)
    ],
    outputs=gr.Textbox(label="Model Response"),
    title=title_with_image,
    description=description_with_style,
    allow_flagging="never"
)


# Launch the application
iface.launch(share=True)