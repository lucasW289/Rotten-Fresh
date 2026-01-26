import boto3
import time
import os

SQS_ENDPOINT = os.getenv(
    "SQS_ENDPOINT",
    "http://localstack.default.svc.cluster.local:4566"
)

sqs = boto3.client(
    "sqs",
    endpoint_url=SQS_ENDPOINT,
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test"
)

def get_queue_url(name):
    return sqs.get_queue_url(QueueName=name)["QueueUrl"]

def run_worker():
    print(f"[*] Worker connected to {SQS_ENDPOINT}")

    # Wait until queues exist
    while True:
        try:
            video_q = get_queue_url("video-queue")
            completion_q = get_queue_url("completion-queue")
            break
        except Exception:
            print("[!] Queues not ready yet, waiting...")
            time.sleep(3)

    print("[✓] Worker ready")

    while True:
        response = sqs.receive_message(
            QueueUrl=video_q,
            WaitTimeSeconds=10,
            MaxNumberOfMessages=1
        )

        msgs = response.get("Messages", [])
        if not msgs:
            continue

        msg = msgs[0]
        body = msg["Body"]
        print(f"[+] Received: {body}")

        if body.startswith("START:"):
            video_id = body.split("START:")[1].strip()
            print(f"[~] Processing {video_id}")
            time.sleep(5)

            sqs.send_message(
                QueueUrl=completion_q,
                MessageBody=f"DONE: {video_id}"
            )
            print(f"[✓] Sent DONE for {video_id}")

        sqs.delete_message(
            QueueUrl=video_q,
            ReceiptHandle=msg["ReceiptHandle"]
        )

if __name__ == "__main__":
    run_worker()
