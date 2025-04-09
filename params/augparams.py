import argparse

def get_augargs():
    augparser = argparse.ArgumentParser(description='is dataaug really useful in VAE')
    augparser.add_argument('-seq_len_times', type=int, default=2, help='vae seq_len_times')
    augparser.add_argument('-learning', type=int, default=1, help='vae seq_len_times')

    augargs = augparser.parse_args()
    augargs.vaeargs = get_vae_params()
    augargs.ganargs = get_gan_params()
    augargs.diffargs = get_diffts_params()
    return augargs
def get_vae_params():
    vaeparser = argparse.ArgumentParser(description='is dataaug really useful in VAE')
    vaeparser.add_argument('-seq_len', type=int, default=0, help='vae windowsize')
    vaeparser.add_argument('-latent_dim', type=int, default=8, help='vae latent_dim')
    vaeparser.add_argument('-reconstruction_wt', type=float, default=3, help='vae latent_dim')
    vaeparser.add_argument('-hidden_layer_sizes', type=list, default=[50, 100, 200], help='vae hidden_layer_sizes')
    vaeparser.add_argument("-batch_size", type=int, default=128, help="批大小")
    vaeparser.add_argument("-trend_poly", type=int, default=3, help="vae trend_poly")
    vaeparser.add_argument('-custom_seas', type=list, default=None, help="vae custom_seas")
    vaeparser.add_argument('-use_residual_conn', type=bool, default=True, help="vae use_residual_conn")
    vaeparser.add_argument('-seq_len_times', type=int, default=1, help='vae seq_len_times')
    vaeparser.add_argument("-vae_epochs", type=int, default=500, help='vae epochs')
    vaeparser.add_argument("-save_dir", type=str, default="/vae", help="last_dir is ./model")
    vaeparser.add_argument("-num_layers", type=int, default=3)
    vaeparser.add_argument("-learning_rate", type=float, default=0.001)
    return vaeparser.parse_args()

def get_gan_params():
    ganparser = argparse.ArgumentParser(description='is dataaug really useful in GAN')
    ganparser.add_argument('-num_layers', type=int, default=3, help='num_layers')
    ganparser.add_argument('-latent_dim', type=int, default=24, help='latent_dim')
    ganparser.add_argument('-learning_rate', type=float, default=1e-3, help='gan lr')
    ganparser.add_argument('-load_checkpoint', type=bool, default=False)
    ganparser.add_argument("-networks_dir", type=str, default="./savedModels/gan")
    ganparser.add_argument("-synth_size", type=int, default=5000)
    ganparser.add_argument("-emb_epochs", type=int, default=500, help='vae epochs')
    ganparser.add_argument("-batch_size", type=int, default=128, help='batch_size')
    ganparser.add_argument("-gamma", type=int, default=1, help='gan gamma')
    ganparser.add_argument('-seq_len_times', type=int, default=1, help='vae seq_len_times')
    ganargs = ganparser.parse_args()
    return ganargs

def get_diffts_params():
    difftsparser = argparse.ArgumentParser(description="is dataug really useful in diffusionTS")
    difftsparser.add_argument('-max_epochs',type=int, default=500, help='max_epochs')
    difftsparser.add_argument('-batch_size', type=int, default=128, help='batch_size')
    difftsparser.add_argument('-gradient_accumulate_every', type=int, default=1, help='gradient_accumulate_every')
    difftsparser.add_argument("-save_dir", type=str, default="/diffusionTS", help="last_dir is ./model")
    difftsparser.add_argument("-base_lr", type=float, default=0.0001, help='base_lr')
    difftsparser.add_argument('-emadecay', type=float, default=0.99,help='emadecay')
    difftsparser.add_argument('-update_interval', type=int, default=10, help='update_interval')


    seq_length: 24
    feature_size: 28
    n_layer_enc: 4
    n_layer_dec: 3
    d_model: 96  # 4 X 24
    timesteps: 1000
    sampling_timesteps: 1000
    loss_type: 'l1'
    beta_schedule: 'cosine'
    n_heads: 4
    mlp_hidden_times: 4
    attn_pd: 0.0
    resid_pd: 0.0
    kernel_size: 1
    padding_size: 0


