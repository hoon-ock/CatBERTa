import torch, os, shutil
from transformers import (RobertaConfig, RobertaModel)
from model.regressors import *

def section_text_integrator(data, col_list):
    '''
    Integrates the text from different sections into one column.
    
    Example:
        data = pd.DataFrame({'text1': ['a', 'b', 'c'], 'text2': ['d', 'e', 'f']})
        col_list = ['text1', 'text2']
        section_text_integrator(data, col_list)
        data = pd.DataFrame({'text1': ['a', 'b', 'c'], 'text2': ['d', 'e', 'f'], 'text': ['a d', 'b e', 'c f']})
    '''
    df = data.copy()
    df['text'] = df[col_list].apply(lambda x: ' '.join(x), axis=1)
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def backbone_wrapper(backbone, head_type):
    '''
    Wrapper function to add a head to the backbone roberta model
    backbone: RobertaModel (base or pretrained)
    head_type: str, type of head to add to the backbone
    '''

    if head_type == "pooler":
        model = PoolerRegressor(backbone) 
    elif head_type == "regressor1":
        model = MyRegressor1(backbone)
    elif head_type == "regressor2":
        model = MyRegressor2(backbone)
    elif head_type == "regressor3":
        model = MyRegressor3(backbone)

    else:
        raise ValueError(f"Unknown model_head: {head_type}") 
    
    return model


def checkpoint_loader(model, checkpoint_path, load_on_roberta=False):
    '''
    Load checkpoint weights into the model.
    
    Args:
        model (RobertaModel): The model (base or with head) to load the checkpoint weights into.
        checkpoint_path (str): Path to the checkpoint file.
        load_on_roberta (bool): If True, load the checkpoint on the Roberta base model.
                               If False, load the checkpoint on the whole model with the head.

    Notes:
        - If `load_on_roberta` is True, the checkpoint should come from pretraining
          and the Roberta base model's state_dict will be used for matching keys.
        - If `load_on_roberta` is False, the checkpoint is loaded onto the whole model
          including the head. Matching keys will be used for transferring the weights.
        - The `strict` parameter of `load_state_dict` is set to False to allow partial loading
          in case of mismatched layers.
    '''
    model_dict = model.state_dict()
    state_dict = torch.load(checkpoint_path)
    if load_on_roberta:
        # this is option is to load the checkpoint on the roberta_model
        # in this case, the checkpoint should be from pretraining
        if 'mlm' in checkpoint_path or 'nsp' in checkpoint_path:
            matching_state_dict = {k.replace('roberta.', ''): v for k, v in state_dict.items() if k.replace('roberta.', '') in model_dict}
        else:
            # matching_state_dict = {k.replace('roberta_model.', ''): v for k, v in state_dict.items() if k.replace('roberta_model.', '') in model_dict}
            matching_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.replace('model.', '') in model_dict}
    else:
        # this is option is to continue training on the whole model with head
        matching_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    if matching_state_dict.keys() == model_dict.keys():
        print('All keys matched!')
    elif len(matching_state_dict.keys()) == 0:

        raise ValueError(f"Unknown model_head: No matching keys!")

    model.load_state_dict(matching_state_dict, strict=False)
