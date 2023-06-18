import torch, os, shutil
from transformers import (RobertaConfig, RobertaModel)
from model.regressors import (MyModel, MyModel2, 
                              MyModel_MLP, MyModel_MLP2,
                              MyModel_AttnHead, 
                              MyModel_ConcatLast4Layers)
from model.classifiers import MultiLabelClassifier

def backbone_wrapper(backbone, head_type):
    '''
    Wrapper function to add a head to the backbone roberta model
    backbone: RobertaModel (base or pretrained)
    head_type: str, type of head to add to the backbone
    '''

    if head_type == "pooler":
        model = MyModel(backbone) 
    elif head_type == "mlp":
        model = MyModel_MLP(backbone)
    elif head_type == "mlp2":
        model = MyModel_MLP2(backbone)
    elif head_type == "attnhead":
        model = MyModel_AttnHead(backbone)
    elif head_type == "concatlayer":
        model = MyModel_ConcatLast4Layers(backbone)
    elif head_type == "multilabel":
        model = MultiLabelClassifier(backbone)
    else:
        raise ValueError(f"Unknown model_head: {head_type}") 
    
    return model

# def save_config_file(original_file_path, copy_folder, copy_file_name):
#     if not os.path.exists(copy_folder):
#         os.makedirs(copy_folder)
#     # './config/ft_config.yaml', 'ft_config.yaml'
#     shutil.copy(original_file_path, os.path.join(copy_folder, copy_file_name))