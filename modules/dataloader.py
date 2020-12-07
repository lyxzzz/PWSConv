from . import dataset

dataset_dict = {}
for name in getattr(dataset, "__all__"):
    dataset_dict[name] = getattr(dataset, name)

def data_loader(config):
    print('==> Preparing data..')
    dataset_cfg = config.pop("dataset")
    dataset_type = dataset_cfg.pop("type")
    if dataset_type in dataset_dict:
        return dataset_dict[dataset_type](**dataset_cfg)
    else:
        raise ValueError(f'dataset_type={dataset_type} does not support.')

    
