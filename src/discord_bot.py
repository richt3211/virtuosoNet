import json
import requests


def sendToDiscord(message):
    webhook_url = 'https://discord.com/api/webhooks/766877169212850178/Ae6jScKombypw_IJ48qskcKFyJA7mFZ-NR9mjWtAn7WbXnGFM5CWXOlJo2zlQKPPMr7b'
    """
    Post a message to discord API via a Webhook.
    """
    payload = {
        "content": message
    }
    headers = {
        'Content-Type': 'application/json',
    }
    response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
    return response

