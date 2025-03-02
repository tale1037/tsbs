import argparse

def get_augargs():
    augparser = argparse.ArgumentParser(description='is dataaug really useful in VAE')
    augparser.add_argument('-seq_len_times', type=int, default=1, help='vae seq_len_times')
    augparser.add_argument('-learning', type=int, default=1, help='vae seq_len_times')

    augargs = augparser.parse_args()
    augargs.vaeargs = get_vae_params()
    augargs.ganargs = get_gan_params()
    return augargs
def get_vae_params():
    vaeparser = argparse.ArgumentParser(description='is dataaug really useful in VAE')
    vaeparser.add_argument('-seq_len', type=int, default=0, help='vae windowsize')
    vaeparser.add_argument('-latent_dim', type=int, default=8, help='vae latent_dim')
    vaeparser.add_argument('-reconstruction_wt', type=float, default=3, help='vae latent_dim')
    vaeparser.add_argument('-hidden_layer_sizes', type=list, default=[50, 100, 200], help='vae hidden_layer_sizes')
    vaeparser.add_argument("-batch_size", type=int, default=128, help="批大小")
    vaeparser.add_argument("-trend_poly", type=int, default=0, help="vae trend_poly")
    vaeparser.add_argument('-custom_seas', type=list, default=None, help="vae custom_seas")
    vaeparser.add_argument('-use_residual_conn', type=bool, default=True, help="vae use_residual_conn")
    vaeparser.add_argument('-seq_len_times', type=int, default=1, help='vae seq_len_times')
    vaeparser.add_argument("-vae_epochs", type=int, default=300, help='vae epochs')
    vaeparser.add_argument("-save_dir", type=str, default="/vae", help="last_dir is ./model")
    vaeparser.add_argument("-num_layers", type=int, default=3)
    vaeparser.add_argument("-learning_rate", type=float, default=0.001)
    return vaeparser.parse_args()

def get_gan_params():
    ganparser = argparse.ArgumentParser(description='is dataaug really useful in GAN')
    ganparser.add_argument('-num_layers', type=int, default=3, help='num_layers')
    ganparser.add_argument('-latent_dim', type=int, default=8, help='latent_dim')
    ganparser.add_argument('-learning_rate', type=float, default=0.001, help='gan lr')
    ganparser.add_argument('-load_checkpoint', type=bool, default=False)
    ganparser.add_argument("-networks_dir", type=str, default="./savedModels/gan")
    ganparser.add_argument("-synth_size", type=int, default=0)
    ganparser.add_argument("-emb_epochs", type=int, default=1000, help='vae epochs')
    ganparser.add_argument("-batch_size", type=int, default=256, help='batch_size')
    ganparser.add_argument("-gamma", type=int, default=1, help='gan gamma')
    ganparser.add_argument('-seq_len_times', type=int, default=1, help='vae seq_len_times')
    ganargs = ganparser.parse_args()
    return ganargs