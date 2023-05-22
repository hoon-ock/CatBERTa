import numpy as np
import pickle
import os
import ase.io
# import openai
# from openai.error import RateLimitError
from tqdm import tqdm
import time


from mp_api.client import MPRester
import mp_api.client.core.client as mp_client
from site_analysis import SiteAnalyzer


class Coord2Language:
    def __init__(self, code_list, tag_path, meta_path, ref_path, ads_path, traj_path, save_path):
        self.code_list = code_list 
        self.tags = pickle.load(open(tag_path,'rb'))
        self.meta = pickle.load(open(meta_path,'rb'))
        self.refE = pickle.load(open(ref_path, 'rb'))
        self.ads_dict = pickle.load(open(ads_path, 'rb'))
        self.traj_path = traj_path
        self.save_path = save_path
        self.mpr = MPRester("AfhjSVXncrif1cc3Zs91RIDahItQpAV6")

    def get_descriptions(self):
        # self.access_openai()
        for code in tqdm(self.code_list):
            save_file_path = os.path.join(self.save_path, f"{code}.pkl")
            if os.path.exists(save_file_path):
                print(f'{code} already exists')
                continue
            # time1 = time.time()
            sys_descr = self.get_sys_description(code)
            # time2 = time.time()
            relaxed_energy = self.get_relaxed_energy(code)
            # time3 = time.time()
            conf_descr_i = self.get_config_description(code, frame_no=0)
            conf_descr_f = self.get_config_description(code, frame_no=-1, relaxed_energy=relaxed_energy)
            # time4 = time.time()
            ads_descr = self.get_ads_description(code)
            # time5 = time.time()
            cat_descr = self.get_cat_description(code)            
            # time6 = time.time()
            results = {'system': sys_descr,
                       'conf_i': conf_descr_i,
                       'conf_f': conf_descr_f,
                       'ads': ads_descr,
                       'cat': cat_descr,
                       'E_ads': relaxed_energy
                       }
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            with open(save_file_path, "wb") as f:
                pickle.dump(results,f)
            # time7 = time.time()
            
            # print('time sys des: ', time2-time1)
            # print('time conf des: ', time4-time3)
            # print('time ads des: ', time4-time3)
            # print('time cat des: ', time5-time4)
            # print('time E calc: ', time3-time2)
            # print('time file save: ', time7-time6)
            # print(conf_descr_i)
            # print(conf_descr_f)

    # def access_openai(self):
    #     os.environ["OPENAI_API_KEY"] = "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"
    #     openai.api_key = os.environ["OPENAI_API_KEY"]

    # def chatgpt(self, prompt):
    #     try:
    #         output = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                     messages=prompt,
    #                                     temperature=1.0,
    #                                     max_tokens=2048,
    #                                     n=1,
    #                                     frequency_penalty=0.0,
    #                                     presence_penalty=0.0)
    #         output_description = output.choices[0].message.content
    #         return output_description
    #     except RateLimitError as e:
    #         print("Got RateLimitError. Retrying in 10 seconds...")
    #         time.sleep(10)
    #         return self.chatgpt(prompt)



    def get_binding_info(self, code, frame_no):
        images = ase.io.read(self.traj_path+code+'.traj', ":")
        tags = self.tags 
        images[frame_no].set_tags(tags[code])
        data = SiteAnalyzer(adslab=images[frame_no])
        return data.binding_info
    
    def get_relaxed_energy(self, code):
        images = ase.io.read(self.traj_path+code+'.traj', ":")
        E_sys = images[-1].get_potential_energy()
        E_ref = self.refE[code]
        E_ads = E_sys - E_ref
        return E_ads

    def get_config_description(self, code, frame_no, relaxed_energy=None):
        config_description = ""
        heavy_atoms = {'C', 'N', 'O'}
        counts = {}
        binding_info = self.get_binding_info(code, frame_no)
        
        if not binding_info:
            return "There is no significant chemical interaction \
between any atoms of the adsorbate and the surface atoms."

        interacting_atoms = [atom for atom in binding_info]
        interacting_heavy_atoms = [atom for atom in binding_info if atom['adsorbate_element'] in heavy_atoms]
            
        if len(interacting_heavy_atoms) >=1:
           interacting_atom_list = interacting_heavy_atoms
           overview = ""
        else:
            interacting_atom_list = interacting_atoms
            overview = "There is no significant chemical interaction between surface \
and heavy atoms in adsorbate. "
        for atom in interacting_atom_list:
            ads = atom['adsorbate_element']
            counts.setdefault(ads, 0)
            counts[ads] += 1
            if counts[ads] == 1:
                prefix = ""
            elif counts[ads] ==2:
                prefix = "2nd"
            elif counts[ads] ==3:
                prefix = "3rd"
            else:
                prefix = f"{counts[ads]}th"        

            cats = ", ".join(str(x) for x in atom['slab_atom_elements'])
            if len(atom['slab_atom_elements'])==1:
                site_type = 'atop'
            elif len(atom['slab_atom_elements'])==2:
                site_type = 'bridge'
            else:
                site_type = 'hollow'
#             description = f"The {prefix} {ads} atom of the adsorbate is placed on the {site_type} site \
# and is binding to the catalytic surface atoms {cats}."
            if prefix:
                description = f"The {prefix} {ads} atom of the adsorbate is placed on the {site_type} site \
and is binding to the catalytic surface atoms {cats}."
            else:
                description = f"The {ads} atom of the adsorbate is placed on the {site_type} site \
and is binding to the catalytic surface atoms {cats}."
            config_description += description

        if frame_no == -1:
            if relaxed_energy:
                E_ads = np.round(relaxed_energy ,3)
            else:
                E_ads = np.round(self.get_relaxed_energy(code),3)
            E_description = f" The calculated adsorption energy of the relaxed structure is {E_ads} eV."
            config_description += E_description
        config_description = overview + config_description
        return config_description
    
    def get_sys_description(self, code):
        meta = self.meta
        #import pdb;pdb.set_trace() 
        sys_description = f"Adsorbate {meta[code]['ads_symbols']} is adsorbed on \
the catalytic surface {meta[code]['bulk_symbols']} ({meta[code]['bulk_mpid']}) \
with a Miller Index of {meta[code]['miller_index']}."
        return sys_description
    
#     def get_ads_description(self, code):
#         meta = self.meta
#         ads = meta[code]['ads_symbols'].replace('*','')
#         init_ads_prompt=[{"role": "system", 
#                     "content": "You give correct and exact available scientific information \
# about adsorbate molecules in the adsorbate-catalyst system, like bonding type, molecule size, \
# bond angle, bond length, orbital characteristics, and dipole moment."},
#                     {"role": "user", 
#                      "content": f"describe the correct and exact relevant chemical information \
# about the adsorbate molecule {ads}, including valid and verifiable information \
# about bonding type, molecule size, bond angle, bond length, \
# orbital characteristics, and dipole moment."},]

#         ads_description = self.chatgpt(init_ads_prompt)
#         return ads_description
    def get_ads_description(self, code):
        meta = self.meta
        ads = meta[code]['ads_symbols']
        ads_description = self.ads_dict[ads]
        return ads_description
    
    def get_cat_description(self, code):
        try:
            mpr = self.mpr
            mpid = self.meta[code]['bulk_mpid']
            try:
                cat_description = mpr.robocrys.get_data_by_id(mpid).description
            #return cat_description
            except:
                cat_description = f"Error occurred while retrieving the record: {mpid}."
                print("Error occurred while retrieving the record:", mpid)
        except mp_client.MPRestError as e:
            
            mpid = self.meta[code]['bulk_mpid']
            cat_description = f"Error occurred while retrieving the record: {mpid}."
            print("Error occurred while retrieving the record:", e)
        return cat_description
        
if __name__ == "__main__":
    import random

    tag_path = '../metadata/oc20_meta/oc20_adslab_tags_full.pkl'
    meta_path = '../metadata/oc20_meta/oc20_data_metadata.pkl'
    ref_path = '../metadata/oc20_meta/oc20_ref_energies.pkl'
    ads_path = '../metadata/prompts/ads_descriptions_mol_updated.pkl'
    traj_path = '/home/jovyan/shared-datasets/OC20/trajs/train_02_01/'
    save_path = '/home/jovyan/shared-scratch/jhoon/CATBERT/train_prompt_dict/'

    code_list = pickle.load(open('../metadata/split_ids/train/train_ids.pkl','rb'))
    # empty_value_ids = ['random864551', 'random230123', 'random1073229',
    #                    'random1335784', 'random938574', 'random1167515']
    seed = 3
    sample_number = 20000
    random.seed(seed)
    sampled_code = random.sample(code_list,sample_number)
    #import pdb;pdb.set_trace()
    convert = Coord2Language(sampled_code, tag_path, meta_path, 
                             ref_path, ads_path, traj_path, save_path)
    convert.get_descriptions()
