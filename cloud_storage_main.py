import os
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "service_account_key.json"
os.environ['BUCKET_NAME'] = "python-ce-bucket"


from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

download_blob(os.environ.get("BUCKET_NAME"), 'realdonaldtrump.csv', 'download.csv')



