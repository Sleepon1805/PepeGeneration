import torch
import numpy as np
from typing import List

""" Funcs to encode-decode conditions """


CELEBA_CONDITION_FEATURES = {
    "5_o_Clock_Shadow": 0,
    "Arched_Eyebrows": 1,
    "Attractive": 2,
    "Bags_Under_Eyes": 3,
    "Bald": 4,
    "Bangs": 5,
    "Big_Lips": 6,
    "Big_Nose": 7,
    "Black_Hair": 8,
    "Blond_Hair": 9,
    "Blurry": 10,
    "Brown_Hair": 11,
    "Bushy_Eyebrows": 12,
    "Chubby": 13,
    "Double_Chin": 14,
    "Eyeglasses": 15,
    "Goatee": 16,
    "Gray_Hair": 17,
    "Heavy_Makeup": 18,
    "High_Cheekbones": 19,
    "Male": 20,
    "Mouth_Slightly_Open": 21,
    "Mustache": 22,
    "Narrow_Eyes": 23,
    "No_Beard": 24,
    "Oval_Face": 25,
    "Pale_Skin": 26,
    "Pointy_Nose": 27,
    "Receding_Hairline": 28,
    "Rosy_Cheeks": 29,
    "Sideburns": 30,
    "Smiling": 31,
    "Straight_Hair": 32,
    "Wavy_Hair": 33,
    "Wearing_Earrings": 34,
    "Wearing_Hat": 35,
    "Wearing_Lipstick": 36,
    "Wearing_Necklace": 37,
    "Wearing_Necktie": 38,
    "Young": 39,
}
CONDITION_SIZE = max(CELEBA_CONDITION_FEATURES.values()) + 1


def encode_condition(dataset_name: str, condition: List[str] | str) -> np.ndarray:
    if dataset_name == 'celeba':
        assert isinstance(condition, list)
        cond_size = (1, CONDITION_SIZE)
        decoded_cond = np.full(cond_size, -1, dtype=np.float32)
        for feature in condition:
            if feature in CELEBA_CONDITION_FEATURES.keys():
                decoded_cond[..., CELEBA_CONDITION_FEATURES[feature]] = 1
            else:
                raise ValueError(f'Given feature {feature} is not a celeba feature.')
    elif dataset_name == 'pepe':
        assert isinstance(condition, str)
        enum_letter = (lambda s: ord(s) - 97)  # enumerate lower case letters from 0 to 25
        decoded_cond = torch.zeros((26, CONDITION_SIZE))  # set length to 40
        for i, letter in enumerate(condition):
            decoded_cond[enum_letter(letter), i] = 1
    else:
        raise ValueError
    return decoded_cond


def decode_condition(dataset_name: str, encoded_cond: torch.Tensor | np.ndarray) -> List[List[str]]:
    # to Tensor
    if isinstance(encoded_cond, np.ndarray):
        encoded_cond = torch.from_numpy(encoded_cond)

    # must be a condition for a single sample
    assert len(encoded_cond.shape) <= 1 + (dataset_name == 'pepe')

    if dataset_name == 'celeba':
        feature_indices = torch.where(encoded_cond == 1)[0]
        features = [feature for feature, index in CELEBA_CONDITION_FEATURES.items() if index in feature_indices]
        decoded_conditions = features
    elif dataset_name == 'pepe':
        raise NotImplementedError   # TODO
    else:
        raise ValueError
    return decoded_conditions
