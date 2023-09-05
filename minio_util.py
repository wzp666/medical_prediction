from xmlrpc.client import ResponseError

import minio
import os
from datetime import timedelta
from minio import Minio

bucket = 'segment-imgs'

def init_minio_client():
    print("init minio client...")
    minioClient = Minio('127.0.0.1:9000',
                        access_key='ee8JgzpPUXTWViZt0Nc3',
                        secret_key='1tCMU6hDEyhm3awwt4vCKsdwXNJII6U3Dq5QQmPZ',
                        secure=False)
    print("init minio client success!")
    return minioClient


def upload_data(minio_client, file_name, file_path, minio_bucket):
    try:
        with open(file_path, 'rb') as file_data:
            file_stat = os.stat(file_path)
            minio_client.put_object(minio_bucket, file_name, file_data,
                                    file_stat.st_size)
    except ResponseError as err:
        print(err)

    try:
        return minio_client.presigned_get_object(minio_bucket, file_name, expires=timedelta(days=2))
    except ResponseError as err:
        print(err)


if __name__ == "__main__":
    client = init_minio_client()
    # url = upload_data(minio_client=client, file_name='1.png', file_path=r'D:\Projects\medical_prediction_py_service\disease_segmentation_version5\dia_output\2181a2f3-4e11-41c7-8d58-3fb43e786808.png')
    # print(url)
