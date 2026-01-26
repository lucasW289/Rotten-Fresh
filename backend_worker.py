import boto3
import json
import time
import os
import signal

# Since you're running on your Mac, use localhost. 
# If running inside K8s, the env var will override this.
LOCALSTACK_URL = os.getenv("SQS_ENDPOINT_URL", "http://localhost:4566")
QUEUE_NAME = "video-render-queue"
REGION = "us-east-1"

sqs = boto3.client('sqs', 
                   endpoint_url=LOCALSTACK_URL,
                   region_name=REGION,
                   aws_access_key_id='test',
                   aws_secret_access_key='test')

def ensure_queue_exists():
    while True:
        try:
            print(f"🔍 Checking for queue at {LOCALSTACK_URL}...")
            response = sqs.create_queue(QueueName=QUEUE_NAME)
            return response['QueueUrl']
        except Exception as e:
            print(f"⏳ Waiting for port-forward/LocalStack... ({e})")
            time.sleep(5)

actual_queue_url = ensure_queue_exists()
keep_running = True

def handle_signal(signum, frame):
    global keep_running
    keep_running = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

print(f"🎧 Worker started on: {actual_queue_url}")

while keep_running:
    try:
        response = sqs.receive_message(
            QueueUrl=actual_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,
            VisibilityTimeout=60 # Prevents other workers from seeing this for 60s
        )

        if 'Messages' in response:
            for msg in response['Messages']:
                body = json.loads(msg['Body'])
                print(f"📨 PROCESSING: {body.get('video_id')}")
                
                time.sleep(2) # Simulating work

                sqs.delete_message(QueueUrl=actual_queue_url, ReceiptHandle=msg['ReceiptHandle'])
                print(f"✅ DONE: {body.get('video_id')}")
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(2)