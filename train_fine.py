import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model, create_model_fine
from util.visualizer import Visualizer_Fine

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)
    # create a model
    fine_model = create_model_fine(opt)
    # create a visualizer
    visualizer = Visualizer_Fine(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter+opt.niter_decay
    total_iteration = opt.iter_count

    # training process
    for epoch in range(1, max_iteration+1):
        epoch_start_time = time.time()
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            fine_model.set_input(data)
            fine_model.optimize_parameters()

            # display images on visdom and save images
            if total_iteration % opt.display_freq == 0:
                visualizer.display_current_results(fine_model.get_current_visuals(), epoch)

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = fine_model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)

            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                fine_model.save_networks('latest')

            # save the model every <save_iter_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0:
                print('saving the model of iterations %d' % total_iteration)
                fine_model.save_networks(total_iteration)

        fine_model.update_learning_rate()

        print('\nEnd training')
