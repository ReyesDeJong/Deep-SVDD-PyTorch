import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from src.utils.config import Config
from src.deepSVDD import DeepSVDD
from src.datasets.main import load_dataset


################################################################################
# Settings
################################################################################
def plot_confusion_matrix(y_true, y_pred, classes,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  if not title:
    if normalize:
      title = 'Normalized confusion matrix %.4f' % (np.trace(cm) / np.sum(cm))
    else:
      title = 'Confusion matrix, without normalization %.4f' % (
          np.trace(cm) / np.sum(cm))

  # Only use the labels that appear in the data
  classes = classes[np.unique(np.stack([y_true, y_pred]))]
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix %.4f' % (np.trace(cm) / np.sum(cm)))
  else:
    print('Confusion matrix, without normalization %.4f' % (
        np.trace(cm) / np.sum(cm)))

  print(cm)

  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         # ... and label them with the respective list entries
         xticklabels=classes, yticklabels=classes,
         title=title,
         ylabel='True label',
         xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, format(cm[i, j], fmt),
              ha="center", va="center",
              color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  plt.show()
  return ax


def plot_threshold_acc(inliers_score, outliers_score, save_path=None,
    plot_show=False):
  thresholds = np.unique(np.concatenate([inliers_score, outliers_score]))
  mean_inliers = np.mean(inliers_score)
  mean_outliers = np.mean(outliers_score)

  accuracies = []
  if mean_outliers < mean_inliers:
    print('out<in')
    for thr in thresholds:
      FP = np.sum(outliers_score >= thr)
      TP = np.sum(inliers_score >= thr)
      TN = np.sum(outliers_score < thr)
      FN = np.sum(inliers_score < thr)

      accuracy = (TP + TN) / (FP + TP + TN + FN)
      accuracies.append(accuracy)

  elif mean_outliers > mean_inliers:
    print('out>in')
    for thr in thresholds[::-1]:
      FP = np.sum(outliers_score < thr)
      TP = np.sum(inliers_score < thr)
      TN = np.sum(outliers_score >= thr)
      FN = np.sum(inliers_score >= thr)

      accuracy = (TP + TN) / (FP + TP + TN + FN)
      accuracies.append(accuracy)

  min = np.min(np.concatenate([inliers_score, outliers_score]))
  max = np.max(np.concatenate([inliers_score, outliers_score]))

  n_bins = 100
  bin_val_inlier, bins_inlier, _ = plt.hist(inliers_score, n_bins,
                                            histtype='step', lw=2,
                                            label='inlier %s' % set,
                                            range=[min, max])
  bin_val_outlier, bins_outlier, _ = plt.hist(outliers_score, n_bins,
                                              histtype='step', lw=2,
                                              label='inlier %s' % set,
                                              range=[min, max])
  max_bin_val = np.max(np.concatenate([bin_val_inlier, bin_val_outlier]))
  plt.close()

  fig, ax = plt.subplots(1, 1)
  ax.step(bins_inlier, np.array([0] + bin_val_inlier.tolist()) / max_bin_val,
          label='inlier', lw=2)
  ax.step(bins_outlier, np.array([0] + bin_val_outlier.tolist()) / max_bin_val,
          label='outlier', lw=2)
  # center = (bins_inlier[:-1] + bins_inlier[1:]) / 2
  # ax.bar(center, bin_val_inlier / max_bin_val, align='center',
  #        width=bins_inlier[1] - bins_inlier[0], label='inlier %s' % set)
  # # print(center, bin_val_inlier/max_bin_val)
  # center = (bins_outlier[:-1] + bins_outlier[1:]) / 2
  # ax.bar(center, bin_val_outlier / max_bin_val, align='center',
  #        width=bins_inlier[1] - bins_inlier[0], label='outlier %s' % set)
  # print(center, bin_val_inlier / max_bin_val)
  ax.plot(thresholds, accuracies, lw=2, label='Accuracy by thresholds',
          color='black')
  ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
  ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

  ax.legend()
  ax.set_ylim([0, 1.0])
  ax.set_xlim([min, max])
  ax.set_ylabel('Accuracy', fontsize=12)
  ax.set_xlabel('SVDD score', fontsize=12)
  ax.grid(ls='--')
  if save_path:
    plt.savefig(os.path.join(save_path, 'thr_acc.png'),
                bbox_inches='tight')
  if plot_show:
    plt.show()


def plot_roc(labels, scores, save_path=None, show_plot=False):
  fpr, tpr, _ = roc_curve(labels, scores)
  auc_value = auc(fpr, tpr)
  fig, ax = plt.subplots()
  lw = 2
  ax.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
  ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('Receiver operating characteristic example')
  ax.legend(loc="lower right")
  if show_plot:
    plt.show()
  if save_path:
    plt.savefig(os.path.join(save_path, 'roc.png'),
                bbox_inches='tight')


if __name__ == '__main__':
  # when epochs = 2, good results

  dataset_name = 'hits'
  net_name = 'hits_LeNet'
  xp_path = '../log/hits_test'
  data_path = '../../datasets/HiTS2013_300k_samples.pkl'
  load_config = None
  load_model = '../log/hits_test/model.tar'
  objective = 'one-class'
  nu = 0.1
  device = 'cuda'
  seed = -1
  optimizer_name = 'adam'
  lr = 0.0001
  n_epochs = 2
  lr_milestone = [50]
  batch_size = 200
  weight_decay = 0.5e-6
  pretrain = True
  ae_optimizer_name = 'adam'
  ae_lr = 0.0001
  ae_n_epochs = 2
  ae_lr_milestone = [50]
  ae_batch_size = 200
  ae_weight_decay = 0.5e-3
  n_jobs_dataloader = 16
  normal_class = 1

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

  # Test model
  deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

  # Plot most anomalous and most normal (within-class) test samples
  indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
  indices, labels, scores = np.array(indices), np.array(labels), np.array(
      scores)

  inliers_score = scores[labels == 0]
  outliers_score = scores[labels == 1]
  plot_threshold_acc(inliers_score, outliers_score, save_path=xp_path)
  plot_roc(labels, scores, save_path=xp_path)


