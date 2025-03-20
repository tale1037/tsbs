import numpy as np
from tqdm import trange

from augMethod.gan import timegan
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
from metrics.visualize import plot_samples
from augMethod.gan.utils import extract_time


def timegantrain(opt, ori_data):

    # Model Setting
    model = timegan.TimeGAN(opt, ori_data)
    #per_print_num = opt.emb_epochs / opt.print_times
    opt.iterations = opt.augargs.ganargs.emb_epochs
    # 1. Embedding network training
    print('Start Embedding Network Training')
    logger = trange(opt.iterations, desc=f"Epoch: 0, Loss: 0")
    for i in logger:
        model.gen_batch()
        model.batch_forward()
        model.train_embedder()

        logger.set_description('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')
    logger = trange(opt.iterations, desc=f"Epoch: 0, Loss: 0")
    for i in logger:
        model.gen_batch()
        model.batch_forward()
        model.train_supervisor()

        logger.set_description('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)))

    # 3. Joint Training
    print('Start Joint Training')
    logger = trange(opt.iterations, desc=f"Epoch: 0, Loss: 0")
    for i in logger:
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            model.gen_batch()
            model.batch_forward()
            model.train_generator(join_train=True)
            model.batch_forward()
            model.train_embedder(join_train=True)
        # Discriminator training
        model.gen_batch()
        model.batch_forward()
        model.train_discriminator()

        # Print multiple checkpoints
        logger.set_description('step: ' + str(i) + '/' + str(opt.iterations) +
                  ', d_loss: ' + str(np.round(model.D_loss.item(), 4)) +
                  ', g_loss_u: ' + str(np.round(model.G_loss_U.item(), 4)) +
                  ', g_loss_s: ' + str(np.round(np.sqrt(model.G_loss_S.item()), 4)) +
                  ', g_loss_v: ' + str(np.round(model.G_loss_V.item(), 4)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(model.E_loss_T0.item()), 4)))
    print('Finish Joint Training')

    # Save trained networks
    model.save_trained_networks(opt.data_name)


def timegantest(opt, ori_data):

    print('Start Testing')
    # Model Setting
    model = timegan.TimeGAN(opt, ori_data)
    model.load_trained_networks(opt.data_name)

    # Synthetic data generation
    if opt.augargs.ganargs.synth_size != 0 and opt.augargs.ganargs.synth_size < len(ori_data):
        synth_size = opt.augargs.ganargs.synth_size
        ori_data1 = ori_data[:synth_size]
    else:
        synth_size = int(len(ori_data))
        ori_data1 = ori_data
    print(ori_data1.shape)
    generated_data = model.gen_synth_data(synth_size,ori_data1)
    generated_data = generated_data.cpu().detach().numpy()
    print(generated_data.shape)
    print(ori_data.shape)
    gen_data = list()
    for i in range(synth_size):
        temp = generated_data[i, :opt.seq_len*opt.augargs.vaeargs.seq_len_times, :]
        gen_data.append(temp)
    print('Finish Synthetic Data Generation')

    # Performance metrics
    metric_results = dict()
    # if not opt.only_visualize_metric:
    #     # 1. Discriminative Score
    #     discriminative_score = list()
    #     print('Start discriminative_score_metrics')
    #     for i in range(opt.metric_iteration):
    #         print('discriminative_score iteration: ', i)
    #         temp_disc = discriminative_score_metrics(ori_data, gen_data)
    #         discriminative_score.append(temp_disc)
    #
    #     metric_results['discriminative'] = np.mean(discriminative_score)
    #     print('Finish discriminative_score_metrics compute')
    #
    #     # 2. Predictive score
    #     predictive_score = list()
    #     print('Start predictive_score_metrics')
    #     for i in range(opt.metric_iteration):
    #         print('predictive_score iteration: ', i)
    #         temp_predict = predictive_score_metrics(ori_data, gen_data)
    #         predictive_score.append(temp_predict)
    #     metric_results['predictive'] = np.mean(predictive_score)
    #     print('Finish predictive_score_metrics compute')

    plot_samples(
        samples1=ori_data,
        samples1_name="Original Train",
        samples2=generated_data,
        samples2_name="Reconstructed Train",
        num_samples=5,
    )
    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, gen_data, 'pca', opt.out_dir + "/dataaug_metrics","timeGAN",opt.data_name)
    visualization(ori_data, gen_data, 'tsne', opt.out_dir,"timeGAN",opt.data_name)


    # Print discriminative and predictive scores
    print(metric_results)
