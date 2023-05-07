import requests
from datetime import datetime
import time

hora = datetime.now()
hora = hora.strftime("%H:%M:%S")

url = 'https://api.dontpad.com/thalistdg'

headers = {}

body = {
    "text": f"Terminou {hora}",
    "captcha-token-v2":	"",
    "force": "true",
}


try:
    req = requests.post(url,headers=headers, params=body)
    print(req)
except Exception as ex:
    # raise ex
    time.sleep(5)
    req = requests.post(url,headers=headers, params=body)
