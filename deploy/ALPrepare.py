
import argparse
import requests

parser = argparse.ArgumentParser(description='Call database to load the data')

parser.add_argument('--bucket', type=str, 
                    default='bucket_name')

parser.add_argument('--directory_name', type=str, 
                    default='Dataset directory')

args = parser.parse_args()

print(f"Calling database to load the AL results, bucket: {args.bucket}, Dataset: {args.directory_name}")

api_url = "http://172.16.1.32:9006/clound/instance/v1/task_prepare"

payload = {
    "bucket" : args.bucket,
    "directoryName" : args.directory_name
}

try:
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        data = response.json()
        if data['data']['size'] > 0 and data['message'] == 'SUCCESS':
            print(f"Calling success, size: {data['data']['size']}.")
except requests.RequestException as e:
    print(f"API调用出现异常: {e}")