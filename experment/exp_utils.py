from augMethod.aug_utils import train_aug_models, generate_aug_data


def trainaugmodel(args,model_names,train_data):
    for model_name in model_names:
        train_aug_models(args,model_name,train_data)


def get_gen_datas(args,model_names,train_data):
    gen_datas = {}
    for model_name in model_names:
        gen_data = generate_aug_data(args,train_data,model_name)
        gen_datas[model_name] = gen_data

    return gen_datas

