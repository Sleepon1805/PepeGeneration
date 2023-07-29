import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def collect_twitch_emotes(data_path):
    """
    API docs: https://www.frankerfacez.com/developers
    Emote search: https://www.frankerfacez.com/emoticons/?q=peepo&sort=count-desc&days=0
    """

    os.makedirs(data_path, exist_ok=True)
    print(f'Saving images as {data_path}')

    base_url = 'https://api.frankerfacez.com/v1/'
    emoticons_url = base_url + 'emoticons'
    params = {
        'page': '1',
        'per_page': '200',
    }
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
            if check_emoticon(emoticon):
                try:
                    size_scale = max(emoticon['urls'].keys(), key=int)
                    url = emoticon['urls'][size_scale]
                    image_request = requests.request('GET', url)
                    image = Image.open(BytesIO(image_request.content))
                    save_path = data_path + emoticon['name'] + '.png'
                    if os.path.exists(save_path):
                        c = 2
                        while os.path.exists(save_path):
                            save_path = data_path + emoticon['name'] + str(c) + '.png'
                            c += 1
                    image.save(save_path)
                except Exception as e:
                    print(f'Error for emoticon {emoticon["name"]}, id={emoticon["id"]}, page {page_num}')
                    print(e)


def check_emoticon(emoticon: dict):
    valid_emote = False

    # usages check
    if emoticon['usage_count'] >= 10:
        valid_emote = True

    # name check
    name = emoticon['name'].lower()
    valid_name = any([
        'pepe' in name,
        'pepo' in name,
        'peepo' in name,
        'monka' in name,
        # name.endswith('ge'),
        # name.endswith('eg'),
    ])
    if valid_name:
        valid_emote = True

    return valid_emote


if __name__ == '__main__':
    path = '/home/sleepon/data/AllFFZEmotes/'
    collect_twitch_emotes(path)

    # import shutil
    # old_path = '/home/sleepon/data/TwitchPepeDatasetv0/'
    # for filename in os.listdir(old_path):
    #     if check_emote_name(filename):
    #         shutil.copy2(old_path + filename, path + filename)
