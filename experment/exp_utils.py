from augMethod.aug_utils import train_aug_models, generate_aug_data
from metrics.visualization_metrics import visualization
from metrics.visualize import plot_samples


def trainaugmodel(args,model_names,train_data):
    for model_name in model_names:
        train_aug_models(args,model_name,train_data)

def get_gen_datas(args,model_names,train_data):
    gen_datas = {}
    for model_name in model_names:
        gen_data = generate_aug_data(args,train_data,model_name)
        gen_datas[model_name] = gen_data
        plot_samples(
            samples1=train_data,
            samples1_name="Original Train",
            samples2=gen_data,
            samples2_name="Reconstructed Train",
            num_samples=5,
            model_name= model_name
        )

        #visualization(train_data,gen_data,"tsne",args.out_dir,model_name,args.data_name)
    return gen_datas

