import base64
from io import BytesIO
import cv2
import numpy as np
import ocr
import json

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        image_base64 = body['image'].encode('utf-8')
        img_b64dec = base64.b64decode(image_base64)
        img_byteIO = BytesIO(img_b64dec)
        img = cv2.imdecode(np.frombuffer(img_byteIO.read(), np.uint8), 1)
        best_eval = ocr.solve(img)
        return {
            'statusCode': 200,
            'headers': {
            'Access-Control-Allow-Methods': 'POST, OPTIONS',  # Allow POST and preflight OPTIONS
            'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(best_eval)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing image',
                'error': str(e)
            })
        }