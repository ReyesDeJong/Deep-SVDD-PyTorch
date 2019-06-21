import logging
import os
import random
import sys

import numpy as np
import torch

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from src.utils.config import Config
from src.utils.visualization.plot_images_grid import plot_images_grid
from src.deepSVDD import DeepSVDD
from src.datasets.main import load_dataset


################################################################################
# Settings
################################################################################

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model,
    objective, nu, device, seed,
    optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
    pretrain, ae_optimizer_name, ae_lr,
    ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
    n_jobs_dataloader, normal_class):
  """
  Deep SVDD, a fully deep method for anomaly detection.

  :arg DATASET_NAME: Name of the dataset to load.
  :arg NET_NAME: Name of the neural network to use.
  :arg XP_PATH: Export path for logging the experiment.
  :arg DATA_PATH: Root path of data.
  """

  # Get configuration
  cfg = Config(locals().copy())

  # Set up logging
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  log_file = xp_path + '/log.txt'
  file_handler = logging.FileHandler(log_file)
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)

  # Print arguments
  logger.info('Log file is %s.' % log_file)
  logger.info('Data path is %s.' % data_path)
  logger.info('Export path is %s.' % xp_path)

  logger.info('Dataset: %s' % dataset_name)
  logger.info('Normal class: %d' % normal_class)
  logger.info('Network: %s' % net_name)

  # If specified, load experiment config from JSON-file
  if load_config:
    cfg.load_config(import_json=load_config)
    logger.info('Loaded configuration from %s.' % load_config)

  # Print configuration
  logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
  logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

  # Set seed
  if cfg.settings['seed'] != -1:
    random.seed(cfg.settings['seed'])
    np.random.seed(cfg.settings['seed'])
    torch.manual_seed(cfg.settings['seed'])
    logger.info('Set seed to %d.' % cfg.settings['seed'])

  # Default device to 'cpu' if cuda is not available
  if not torch.cuda.is_available():
    device = 'cpu'
  logger.info('Computation device: %s' % device)
  logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

  # Load data
  dataset = load_dataset(dataset_name, data_path, normal_class)

  # Initialize DeepSVDD model and set neural network \phi
  deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
  deep_SVDD.set_network(net_name)
  # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
  if load_model:
    deep_SVDD.load_model(model_path=load_model, load_ae=True)
    logger.info('Loading model from %s.' % load_model)

  logger.info('Pretraining: %s' % pretrain)
  if pretrain:
    # Log pretraining details
    logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
    logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
    logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
    logger.info('Pretraining learning rate scheduler milestones: %s' % (
      cfg.settings['ae_lr_milestone'],))
    logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
    logger.info(
        'Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

    # Pretrain model on dataset (via autoencoder)
    deep_SVDD.pretrain(dataset,
                       optimizer_name=cfg.settings['ae_optimizer_name'],
                       lr=cfg.settings['ae_lr'],
                       n_epochs=cfg.settings['ae_n_epochs'],
                       lr_milestones=cfg.settings['ae_lr_milestone'],
                       batch_size=cfg.settings['ae_batch_size'],
                       weight_decay=cfg.settings['ae_weight_decay'],
                       device=device,
                       n_jobs_dataloader=n_jobs_dataloader)

  # Log training details
  logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
  logger.info('Training learning rate: %g' % cfg.settings['lr'])
  logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
  logger.info('Training learning rate scheduler milestones: %s' % (
    cfg.settings['lr_milestone'],))
  logger.info('Training batch size: %d' % cfg.settings['batch_size'])
  logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

  # Train model on dataset
  deep_SVDD.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

  # Test model
  deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

  # Plot most anomalous and most normal (within-class) test samples
  indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
  indices, labels, scores = np.array(indices), np.array(labels), np.array(
      scores)
  idx_sorted = indices[labels == 0][np.argsort(
      scores[labels == 0])]  # sorted from lowest to highest anomaly score

  # Save results, model, and configuration
  deep_SVDD.save_results(export_json=xp_path + '/results.json')
  deep_SVDD.save_model(export_model=xp_path + '/model.tar')
  cfg.save_config(export_json=xp_path + '/config.json')

  if dataset_name in ('hits', 'mnist', 'cifar10'):

    if dataset_name == 'hits':
      X_normals = torch.tensor(
        np.transpose(dataset.test_set.image_arr[idx_sorted[:32], ...],
                     (0, 3, 1, 2)))
      X_outliers = torch.tensor(
        np.transpose(dataset.test_set.image_arr[idx_sorted[-32:], ...],
                     (0, 3, 1, 2)))

    if dataset_name == 'mnist':
      X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
      X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(
          1)

    if dataset_name == 'cifar10':
      X_normals = torch.tensor(
          np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...],
                       (0, 3, 1, 2)))
      X_outliers = torch.tensor(
          np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...],
                       (0, 3, 1, 2)))

    plot_images_grid(X_normals, export_img=xp_path + '/normals',
                     title='Most normal examples', padding=2)
    plot_images_grid(X_outliers, export_img=xp_path + '/outliers',
                     title='Most anomalous examples', padding=2)


if __name__ == '__main__':
  #when epochs = 2, good results
  params = {
    'dataset_name': 'hits',
    'net_name': 'hits_LeNet',
    'xp_path': '../log/hits_test',
    'data_path': '../../datasets/HiTS2013_300k_samples.pkl',
    'load_config': None,
    'load_model': None,
    'objective': 'soft-boundary',
    'nu': 0.1,
    'device': 'cuda',
    'seed': -1,
    'optimizer_name': 'adam',
    'lr': 0.0001,
    'n_epochs': 150,
    'lr_milestone': [50],
    'batch_size': 200,
    'weight_decay': 0.5e-6,
    'pretrain': True,
    'ae_optimizer_name': 'adam',
    'ae_lr': 0.0001,
    'ae_n_epochs': 150,
    'ae_lr_milestone': [50],
    'ae_batch_size': 200,
    'ae_weight_decay': 0.5e-3,
    'n_jobs_dataloader': 16,
    'normal_class': 1,
  }
  main(**params)
