import pickle
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset,random_split
from Datasets import *
from Models import *
from func_utils import *

# this script is used to open the rf_crate model and get the attention map and feature map #########################################################

model_weights_name = 'ablation_widar3G6_rfcrate_vanilla_rf_crate_small_0116105758.pth'
pretrained_model_path = 'weights/ablation_widar3G6_previous/'+model_weights_name
config_path = 'Configurations/ablation_widar3G6_previous/ablation_widar3G6_rfcrate_vanilla.yaml'
save_indomain_result_flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = str('2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if save_indomain_result_flag:
    result_save_path = 'open_rfcrate_box_results/'+model_weights_name[:-4] + '_indomain_results.pkl'
else:
    result_save_path = 'open_rfcrate_box_results/'+model_weights_name[:-4] + '_cross_domain_results.pkl'

if not os.path.exists('open_rfcrate_box_results'):
    os.makedirs('open_rfcrate_box_results')

with open(config_path, 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)

for key, value in config.items():
        if value == 'None':
            config[key] = None

rng_generator = torch.manual_seed(config['init_rand_seed'])
torch.cuda.manual_seed(config['init_rand_seed'])
np.random.seed(config['init_rand_seed'])

# configur the dataset and dataloader  ########################################
if config['dataset_name'] == 'widar3':
    data_sahpe_coverter = widar3_data_shape_converter(config)
    dataset_get = Get_Widar3_Dataset
    dataloader_make = make_widar3_dataloader
elif config['dataset_name'] == 'widar_gait':
    data_sahpe_coverter = widarGait_data_shape_converter(config)
    dataset_get = WidarGait_Dataset
    dataloader_make = make_widar_gait_dataloader
elif config['dataset_name'] == 'HuPR':  
    data_sahpe_coverter = HuPR_data_shape_converter(config)
    dataset_get = HuPR_Dataset
    dataloader_make = make_HuPR_dataloader
elif config['dataset_name'] == 'OPERAnet_UWB':
    data_sahpe_coverter = OPERAnet_UWB_data_shape_converter(config)
    dataset_get = OPERAnet_UWB_Dataset
    dataloader_make = make_OPERAnet_UWB_dataloader
elif config['dataset_name'] == 'OctoNetMini':
    data_sahpe_coverter = OctonetMini_data_shape_converter(config)
    dataset_get = OctonetMini
    dataloader_make = make_OctonetMini_dataloader
else:
    print("The dataset name is wrong!")
    
cross_domain = False
if len(config['data_split']) ==3:
    print("The normal indomain experiments!")
    train_ratio = config['data_split'][0]
    val_ratio = config['data_split'][1]
    test_ratio = config['data_split'][2]
    merged_config = {**config, **config['all_dataset']} 
    dataset = dataset_get(merged_config)
    train_num = int(len(dataset)*train_ratio)
    val_num = int(len(dataset)*val_ratio)
    test_num = int(len(dataset)) - train_num - val_num
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num], generator=rng_generator)
else:  # used for cross domain experiments
    print("The cross domain experiments!")
    train_set_config = {**config, **config['train_dataset']} 
    dataset = dataset_get(train_set_config)
    if config['dataset_name'] == 'widar3':
        # we need to lable_mapping for the cross domain experiments
        label_mapping = dataset.get_label_mapping()
        print("The label mapping is: ",label_mapping)
        config['label_mapping'] = label_mapping  # save the label mapping for the cross domain experiments
    val_num = int(len(dataset)*(0.2))
    indomain_test_num = int(len(dataset)*(0.2))  # this is used for the in-domain test
    train_num = len(dataset) - val_num - indomain_test_num
    train_set, val_set, indomain_test_set = torch.utils.data.random_split(dataset, [train_num, val_num, indomain_test_num], generator=rng_generator)
    print("finish the train, val, and indomain test loading")
    cross_domain = True


if save_indomain_result_flag:
    test_loader = dataloader_make(indomain_test_set, is_training=False, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])
    del train_set, val_set, indomain_test_set
else:
    test_set_config = {**config, **config['test_dataset']} 
    test_set = dataset_get(test_set_config)
    test_loader = dataloader_make(test_set, is_training=False, generator=rng_generator, batch_size=config['batch_size'],collate_fn_padd=None, num_workers=config['num_workers'])


model_name = config['model_name']
model = get_registered_models(model_name, config)
model.to(device)

state_dict = torch.load(pretrained_model_path )
model.load_state_dict(state_dict, strict=False)

try:
    subspace_regularization = config['ssr']
    regularizer_lambda = config['ssr_lambda']
    if subspace_regularization == True:
        print("The subspace regularization is used")
        if config['model_name'] == 'rf_crate_tiny':
            regularizer = Subspace_Regularization(num_subspace = 6, dim = 384)
        elif config['model_name'] == 'rf_crate_small':
            regularizer = Subspace_Regularization(num_subspace = 12, dim = 576)
        elif config['model_name'] == 'rf_crate_base':
            regularizer = Subspace_Regularization(num_subspace = 12, dim = 768)
        elif config['model_name'] == 'rf_crate_large':
            regularizer = Subspace_Regularization(num_subspace = 16, dim = 1024)
        else:
            raise NotImplementedError
        regularizer.to(device)
    else:
        regularizer = None
        regularizer_lambda = 0
except:
    regularizer = None
    regularizer_lambda = 0

# the rotary physics prior embedding for rf_crate model
try:
    if config['ppe'] is not None:
        print("The rotary physics prior embedding is used")
        rotary_physcis_prior_embedding = RotaryPhysicPriorEmbedding(config, device)
        rotary_physcis_prior_embedding.to(device)
    else:
        rotary_physcis_prior_embedding = None
except:
    rotary_physcis_prior_embedding = None
    


def forward_attn_feature(model,data_loader, config, device, data_sahpe_coverter, rotary_physcis_prior_embedding, layer: int = 11) -> torch.Tensor:
    results = {
            'input': None,
            'attentions': None,
            'qkv': None,
            'feature': None,
            'label': None,
            'attr': None,
        }
    input_list = []
    attentions_list = []
    qkv_list = []
    feature_list = []
    label_list = []
    attr_list = []
    
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, sample in enumerate(tqdm(data_loader)):
            data, label, *attr = sample

            if data_sahpe_coverter is not None:
                if rotary_physcis_prior_embedding is not None:
                    data = rotary_physcis_prior_embedding(data)
                input = data_sahpe_coverter.shape_convert(data)
            else:
                input = data
            if config['format'] == 'complex':
                input = input.cfloat().to(device)
            elif (config['format'] == 'dfs' or config['format'] == 'dense_dfs' or config['format'] == 'dense_dfs_amp') and config['model_input_shape'] == 'BCHW-C':
                input = input.cfloat().to(device)
            else:
                input = input.float().to(device)   

            output = model(input)
            attentions = model.get_selfattention(rf_data = input, layer=layer)
            predicts, feature_pre = output
            qkv = model.get_qkv(rf_data = input, layer=layer)

            attentions_cpu = attentions.cpu().numpy()
            qkv_cpu = qkv.cpu().numpy()
            embedding_cpu = feature_pre.cpu().numpy()
            input_cpu = input.cpu().numpy()
            label_cpu = label.cpu().numpy()
            attr_cpu = [a.cpu().numpy() for a in attr]

            input_list.append(input_cpu)
            attentions_list.append(attentions_cpu)
            qkv_list.append(qkv_cpu)
            feature_list.append(embedding_cpu)
            label_list.append(label_cpu)
            attr_list.append(attr_cpu)
            
            # if arrive ten percent, save the results
            if i == (len(data_loader) // 10):
                results['input'] = np.concatenate(input_list, axis=0)
                results['attentions'] = np.concatenate(attentions_list, axis=0)
                results['qkv'] = np.concatenate(qkv_list, axis=0)
                results['feature'] = np.concatenate(feature_list, axis=0)
                results['label'] = np.concatenate(label_list, axis=0)
                results['attr'] = np.concatenate(attr_list, axis=0)
                return results
        results['input'] = np.concatenate(input_list, axis=0)   
        results['attentions'] = np.concatenate(attentions_list, axis=0)
        results['qkv'] = np.concatenate(qkv_list, axis=0)
        results['feature'] = np.concatenate(feature_list, axis=0)
        results['label'] = np.concatenate(label_list, axis=0)
        results['attr'] = np.concatenate(attr_list, axis=0)
        return results

results = forward_attn_feature(model,test_loader, config, device, data_sahpe_coverter, rotary_physcis_prior_embedding, layer=11)

# save the results
with open(result_save_path, 'wb') as f:
    pickle.dump(results, f)

# # load the results
# with open(result_save_path, 'rb') as f:
#     results = pickle.load(f)