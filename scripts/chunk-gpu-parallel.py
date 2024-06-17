import os
import pandas as pd
import sys
import time
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from multiprocessing import Process, current_process
import multiprocessing

hf_token = os.getenv("HF_TOKEN")






if __name__ == '__main__':
    
    file = "/mnt/ragtest/sec-data/sec-filings-test/2010q2_notes_txt.txt"
    save_path = "/mnt/ragtest/sec-data/sec-filings-embeddings-test"

    counter = time.time()

    model_name="sentence-transformers/all-mpnet-base-v2"
    tokens_per_chunk=350
    chunk_overlap=125
    print("Creating model")
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name)

    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap,
    )

    print(f'{current_process().name} processing file: {file.split("/")[-1]}')

    # Load data
    print(f'Loading data for {file.split("/")[-1]}')
    start_time = time.time()
    loader = UnstructuredFileLoader(file)
    data = loader.load()
    load_time = time.time() - start_time

    # Chunk data
    print(f'Chunking data for {file.split("/")[-1]}')
    start_time = time.time()
    documents = text_splitter.split_documents(data)
    chunk_time = time.time() - start_time
    text_docs = [doc.page_content for doc in documents]

    # Create embeddings
    print(f'Creating embeddings for {file.split("/")[-1]}')
    start_time = time.time()
#    embeddings = model.encode(text_docs)
    pool = model.start_multi_process_pool()

    # Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(text_docs, pool)
    #print("Embeddings computed. Shape:", embeddings.shape)

    # Optional: Stop the processes in the pool
    model.stop_multi_process_pool(pool)

    embed_time = time.time() - start_time
    total_vectors = embeddings.shape[0]

    # Save parquet file
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    df = pd.DataFrame({
        'text': text_docs,
        'embeddings': embeddings_list
    })

    total_time = load_time + chunk_time + embed_time

    new_filename = os.path.basename(file).split("_txt")[0] + "_embeddings.parquet"
    new_path = os.path.join(save_path, new_filename)
    df.to_parquet(new_path)

    file_stats = os.stat(file)

    print(f"""
finished processing {file.split('/')[-1]}
file size: {file_stats.st_size / (1024 * 1024):.1f} MB
load time: {load_time/60:.2f}
chunk time: {chunk_time/60:.2f}
embed time: {embed_time/60:.2f}
total time: {total_time/60:.2f}""")

    time_str = f"All  files processed. Total run time: {(time.time() - counter) / 3600:.2f} hours\n"
    print(time_str)

