import streamlit as st
from tqdm import tqdm
import logging.config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

plt.style.use('seaborn')


def draw_sample(stat, name, epoch, streamlit):

    epoches = np.arange(stat.shape[0])
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

    ax[0].plot(epoches, stat['train_loss'].values, label='train')
    ax[0].plot(epoches, stat['val_loss'].values, label='val')
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()
    ax[1].plot(epoches, stat['train_metric'].values, label='train')
    ax[1].plot(epoches, stat['val_metric'].values, label='val')
    ax[1].set_ylabel('metric')
    ax[1].set_xlabel('epoch')
    ax[1].legend()

    plt.savefig(f'assets/{name}/imgs/{epoch}.png')

    if streamlit:
        return fig
    else:
        plt.cla()
        plt.close(fig)


def experiment(odeint,
               func,
               dataloader,
               val,
               optimizer,
               criterion,
               metric,
               settings,
               log_config,
               streamlit=False):

    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__name__)

    logger.info('Start experiment')
    logger.info(settings)

    if streamlit:
        method = settings['method']
        name = settings['name']
        interval = settings['interval']
        num_epoch = settings['num_epoch']
        st.header('Progress bar')
        pbar = st.progress(0)
        placeholder = st.empty()
    else:
        method = settings.method
        name = settings.name
        interval = settings.interval
        num_epoch = settings.num_epoch
        name = settings.name
        pbar = tqdm(range(num_epoch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = pd.DataFrame([], columns=['train_loss', 'val_loss', 'train_metric', 'val_metric'])

    with pbar:

        for epoch in range(num_epoch):

            run_loss = []
            run_metric = [] #?
            val_loss = []
            val_metric = []

            func.train()

            for batch in dataloader:

                optimizer.zero_grad()
                v, z = batch
                v, z = v.to(device), z.to(device)
                z0, z1, t = z[:, 0], z[:, 1], torch.linspace(0, 1, 81, device=device)

                preds = odeint(func, z0, v, t, method=method)[-1]
                loss = criterion(preds, z1)
                loss.backward()
                optimizer.step()

                run_loss.append(loss.item())
                run_metric.append(metric(preds, z1).item())

            """
                Validation
            """


            if not epoch % interval:

                func.eval()

                for batch in val:

                    optimizer.zero_grad()
                    v, z = batch
                    v, z = v.to(device), z.to(device)
                    z0, z1, t = z[:, 0], z[:, 1], torch.linspace(0, 1, 81, device=device)

                    preds = odeint(func, z0, v, t, method=method)[-1]
                    loss = criterion(preds, z1)

                    val_loss.append(loss.item())
                    val_metric.append(metric(preds, z1).item())

                logger.info('Epoch %i\t–\tTrain loss: %.6f\t–\tTrain metric: %.6f\t–\tVal loss: %.6f\t–\tVal metric: %.6f' % (
                    epoch, np.mean(run_loss), np.mean(run_metric), np.mean(val_loss), np.mean(val_metric)))

                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': func.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
                torch.save(state_dict, f'assets/{name}/model.pt')

                stats.loc[epoch, 'train_loss'] = np.mean(run_loss)
                stats.loc[epoch, 'val_loss'] = np.mean(val_loss)
                stats.loc[epoch, 'train_metric'] = np.mean(run_metric)
                stats.loc[epoch, 'val_metric'] = np.mean(val_metric)

            if streamlit:

                if epoch == 0:
                    dm1, dm2, dm3, dm4 = np.zeros(4)
                else:
                    dm1, dm2, dm3, dm4 = np.round(stats.iloc[-1].fillna(0).to_numpy()-stats.iloc[-2].fillna(0).to_numpy(), 3)
                    plt.close(fig)

                placeholder.empty()
                pr_point = round((epoch+1)/num_epoch*100)
                pbar.progress(pr_point)

                with placeholder.container():

                    st.header('Metrix')
                    col1, col2, col3, col4 = st.columns(4)

                    m1, m2, m3, m4 = np.round(stats.iloc[-1].fillna(0).to_numpy(), 3)

                    col1.metric(f"{settings['loss']} train", m1, dm1, delta_color="inverse")
                    col2.metric(f"{settings['loss']} val", m2, dm2, delta_color="inverse")
                    col3.metric(f"{settings['metric']} train", m3, dm3, delta_color="off")
                    col4.metric(f"{settings['metric']} val", m4, dm4, delta_color="off")

                    st.header('Plots')
                    fig = draw_sample(stats, name, epoch, streamlit)
                    st.pyplot(fig, clear_figure=True)

            else:
                pbar.update(1)
                draw_sample(stats, name, epoch, streamlit)

            stats.to_csv(f'assets/{name}/stats.csv')

            state_dict = {
                'epoch': epoch,
                'model_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            torch.save(state_dict, f'assets/{name}/model.pt')

    """
    if streamlit:
        st.success('Experiment finished!')
        st.balloons()
    """
    logger.info('END experiment')
