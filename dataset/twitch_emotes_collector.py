import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def collect_twitch_emotes(config):
    """
    API docs: https://www.frankerfacez.com/developers
    """

    os.makedirs(config.twitch_emotes_data_path, exist_ok=True)
    print(f'Saving images as {config.twitch_emotes_data_path}')

    base_url = 'https://api.frankerfacez.com/v1/'
    emoticons_url = base_url + 'emoticons'
    params = {'page': '1', 'per_page': '200', 'q': 'pepe'}
    req = requests.request('GET', emoticons_url, params=params).json()
    print(f'Pages: {req["_pages"]}, Total: {req["_total"]}')

    num_pages = req["_pages"]
    for page_num in tqdm(range(1, num_pages+1)):
        params['page'] = page_num
        request = requests.request('GET', emoticons_url, params=params)
        req = request.json()
        if 'error' in req.keys():
            print(f'Error in emoticons page request: {request.url}:')
            print(f'Error: {req["error"]}')
            print(f'Message: {req["message"]}')
            continue

        for emoticon in req['emoticons']:
            try:
                size_scale = max(emoticon['urls'].keys(), key=int)
                url = emoticon['urls'][size_scale]
                image_request = requests.request('GET', url)
                image = Image.open(BytesIO(image_request.content))
                save_path = config.twitch_emotes_data_path + emoticon['name'] + '.png'
                if os.path.exists(save_path):
                    c = 2
                    while os.path.exists(save_path):
                        save_path = config.twitch_emotes_data_path + emoticon['name'] + str(c) + '.png'
                        c += 1
                image.save(save_path)
            except Exception as e:
                print(f'Error for emoticon {emoticon["name"]}, id={emoticon["id"]}, page {page_num}')
                print(e)
