from pymilvus import CollectionSchema, FieldSchema, DataType, utility, connections, Collection, list_collections
import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from minio import Minio
from minio.error import S3Error
from tabulate import tabulate

DIM = 768 #embedding size

DEFAULT_BUCKET_NAME = "bucket1"
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"
collection_name = "SEC_Filings"

object_store = sys.argv[2] if len(sys.argv) > 2 else "s3"

batch = sys.argv[1] if len(sys.argv) > 1 else ""

# get year from user
if batch == "":
    print('Please enter a valid batch folder to process files')
    sys.exit()
    
else:
    print(f'Processing files for {batch}')


if object_store == "minio":
    # using default MinIO setup
    pass

else:
    # s500/s3 - please use your own bucket
    config = yaml.safe_load(open("/root/rag-bench/zilliz/k8s/s3-milvus.yaml"))
    DEFAULT_BUCKET_NAME = config["externalS3"]["bucketName"]
    MINIO_ADDRESS = config["externalS3"]["host"] + ":" + str(config["externalS3"]["port"])
    MINIO_SECRET_KEY = config['externalS3']['secretKey']
    MINIO_ACCESS_KEY = config['externalS3']['accessKey']


print(DEFAULT_BUCKET_NAME)
print(MINIO_ADDRESS)
def find_folders_by_year(base_directory: str, year: str) -> list:
    """Find all directories within the base directory that start with the given year."""
    matched_folders = []
    for root, dirs, _ in os.walk(base_directory):
        for dir_name in dirs:
            if dir_name.startswith(year):
                # Construct the full path to the directory
                dir_path = os.path.join(root, dir_name)
                matched_folders.append(dir_path)
    return matched_folders

# get shape of file for ingesting
def get_base_shape(file_path):
    base_data = np.load(file_path, allow_pickle=True)
    return base_data.shape

# upload function to milvus_bulkinsert
# each file must match schema name (embeddings, text, id)
def upload(data_folder: str, bucket_name: str=DEFAULT_BUCKET_NAME)->(bool, list):
    if not os.path.exists(data_folder):
        print("Data path '{}' doesn't exist".format(data_folder))
        return False, []

    remote_files = []
    try:
        print("Prepare upload files")
        minio_client = Minio(endpoint=MINIO_ADDRESS, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        found = minio_client.bucket_exists(bucket_name)
        if not found:
            print(f"MinIO bucket '{bucket_name}' doesn't exist")
            return False, []

        remote_data_path = "milvus_bulkinsert"
        def upload_files(folder:str):
            for parent, dirnames, filenames in os.walk(folder):
                if parent is folder:
                    for filename in filenames:
                        ext = os.path.splitext(filename)
                        if len(ext) != 2 or (ext[1] != ".json" and ext[1] != ".npy"):
                            continue
                        local_full_path = os.path.join(parent, filename)
                        minio_file_path = os.path.join(remote_data_path, os.path.basename(folder), filename)
                        minio_client.fput_object(bucket_name, minio_file_path, local_full_path)
                        print(f"Upload file '{local_full_path}' to '{minio_file_path}'")
                        remote_files.append(minio_file_path)
                    for dir in dirnames:
                        upload_files(os.path.join(parent, dir))

        upload_files(data_folder)

    except S3Error as e:
        print(f"Failed to connect MinIO server {MINIO_ADDRESS}, error: {e}")
        return False, []

    print(f"Successfully upload files: {remote_files}")
    return True, remote_files

# Function that checks if indexing is complete
def wait_index():
    while True:
        progress = utility.index_building_progress(collection_name)
        print(progress)
        if progress.get("pending_index_rows", -1) == 0:
            break
        time.sleep(20)

# Function for pretty printing
def percentage(part, total):
    return round(100 * float(part)/float(total))

# Milvus status codes for Bulk Insert. Need it because they fail silently
task_states = {
    0: "Pending",
    1: "Failed",
    2: "Started",
    5: "Persisted",
    6: "Completed",
    7: "Failed and Cleaned"
}



# Main script logic
def main():
    connections.connect(host="10.233.19.184", port=19530)
    if collection_name not in list_collections():
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=50),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=400),
            FieldSchema(name="categories", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=1000),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
        ]
        schema = CollectionSchema(fields)
        collection = Collection(name=collection_name, schema=schema)
        print(f"Created collection '{collection_name}'.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    # Modify this path to the directory containing your data folders
    data_folder = f'json-path/{batch}'

    
    task_ids = {}
    num_vec = 0


    ok, remote_files = upload(data_folder)

    if not ok:
        print("Some files failed to upload.")
    else:
        print("All files uploaded successfully.")

    print("do_bulk_insert")
    print(len(remote_files))
    for f in remote_files:
        file = []
        file.append(f)
        print(type(file))
        print(file)
        task_id = utility.do_bulk_insert(collection_name=collection_name, files=file)
        print(task_id)

        task_ids[f] = task_id
    print(task_ids)
    #print(f"Total embeddings of processed files: {num_vec:,}")

    # get status of all upload tasks
    all_completed = False

    print("Begin Ingestion")
    # continue to check status every few mins
    begin_t = time.time()
    t = 0
    while not all_completed:
        if t % 60 == 0:
            print(f"Collection size thus far: {collection.num_entities}\n")

            for task_key, task_val in task_ids.items():
                # get state of task (pending, completed, failed, etc)
                # print(utility.get_bulk_insert_state(task_val))
                state = utility.get_bulk_insert_state(task_val).state
                # print(state)
                # print(f"Task: {task_key}\tStatus: {task_states[state]}")

                # if task failed, print why
                if state == 1 or state == 7:
                    reason = utility.get_bulk_insert_state(task_val).failed_reason
                    print(f"{task_key} has failed with reason: {reason}")

        # check status of ids
        #all_completed = all((status == 'Completed' or status == 'Failed') for status in task_ids.values())

        all_completed = all((utility.get_bulk_insert_state(status).state_name == 'Completed' or utility.get_bulk_insert_state(status).state_name == 'Failed') for status in task_ids.values())
        time.sleep(1)
        t += 1

    else:
        print('Finished ingestion!')
        print(f"Total Vectors in {collection_name}: {collection.num_entities}")

        for task_key, task_val in task_ids.items():
            # get state of task (pending, completed, failed, etc)
            state = utility.get_bulk_insert_state(task_val).state
            # print(f"Task: {task_key}\tStatus: {task_states[state]}")

            # if task failed, print why
            if state == 1 or state == 7:
                reason = utility.get_bulk_insert_state(task_val).failed_reason
                print(f"{task_key} has failed with reason: {reason}")

    insert_t  = time.time()
    blk_insert_t = insert_t - begin_t
    print(f"Time spent inserting: {blk_insert_t:.2f}\n")


    print("Create Index (~12-14 mins on s3)")
    try:
        collection.create_index(field_name="embeddings",
            index_params = {
              "metric_type":"L2",
              "index_type":"IVF_FLAT",
              "params":{"nlist":1024}
            })
    except Exception as e:
        print(f"index error: {e}")
        raise e from None

    print("wait for indexing to finish: checks for updates every 20 seconds")
    wait_index()
    index_t  = time.time()
    create_index_t  = index_t - insert_t
    print(f"Time indexing: {create_index_t:.2f}\n")

    print("Loading index (1-2 mins)")
    collection.load()

    load_t = time.time()

    load_index_t  = load_t - index_t
    print(f"Time loading: {load_index_t:.2f}")
    total_t = load_t - begin_t

    time_str = f"All  files processed. Total run time: {(total_t) / 60:.2f} mins"
    print(time_str)
    log_str= f"Insert: {blk_insert_t / 60:.2f} mins\tCreate Index: {create_index_t / 60:.2f} mins\tLoad Index: {load_index_t / 60:.2f} mins\tTotal time: {total_t / 60:.2f} mins\n"
    log_file = open('milvus_ingest_time_logs.txt', 'a')
    log_file.write(log_str)
    log_file.close()


if __name__ == "__main__":
    main()
