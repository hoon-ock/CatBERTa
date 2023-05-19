import pickle
import time
import os
import openai
from openai.error import RateLimitError
import tqdm

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

def chatgpt(prompt):
    try:
        output = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                    messages=prompt,
                                    temperature=1.0,
                                    max_tokens=2048,
                                    n=1,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.0)
        output_description = output.choices[0].message.content
        return output_description
    except RateLimitError as e:
        print("Got RateLimitError. Retrying in 10 seconds...")
        time.sleep(10)
        return chatgpt(prompt)


def get_ads_description(ads):
    chem_comp = ads.replace('*','')
    init_ads_prompt=[{"role": "system", 
                    "content": "You give correct and exact available scientific information \
about adsorbate molecules in the adsorbate-catalyst system, like bonding type, molecule size, \
bond angle, bond length, orbital characteristics, and dipole moment."},
                    {"role": "user", 
                     "content": f"Provide accurate and complete chemical information \
about the adsorbate {chem_comp} molecule (or atom or radical). \
This should include details about the bonding type, molecular size, bond angles \
and lengths, orbital characteristics, and dipole moment, all of which must be valid and verifiable."},]

    ads_description = chatgpt(init_ads_prompt)
    return ads_description

if __name__ == '__main__':

    meta_path = '../metadata/oc20_meta/oc20_data_metadata.pkl'
    meta = pickle.load(open(meta_path, 'rb'))

    ads_types = {}
    for code in meta:
        id = meta[code]['ads_id']
        ads = meta[code]['ads_symbols']
        ads_types.update({id:ads})

    ads_list = list(ads_types.values())

    save_file = {}
    for ads in tqdm.tqdm(ads_list):
        ads_description = get_ads_description(ads)
        save_file.update({ads:ads_description})

    with open('../metadata/prompts/ads_descriptions_mol.pkl','wb') as f:
        pickle.dump(save_file, f)