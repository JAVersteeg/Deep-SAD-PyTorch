import math
import torch
import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from datetime import datetime
from PIL import Image
import logging
from time import time

def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """ Plot 4D Tensor of images of shape (B x C x H x W) as a grid. """

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.figure(figsize=(50,50))
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)
    
    img = Image.new(mode='L', size=(4096,4096))
    filename = export_img + '_' + str(datetime.now()) + '.png'
    img.save(filename, "PNG")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    
    
def round_scores(x, num=5):
    return round(x, num)
    
    
def digitize_scores(norms: list, anoms: list, num_bins = 21):
    logger = logging.getLogger()
    
    norm_start = time()
    data = np.append(anoms, norms)
    
    anoms_log = list(map(lambda x: math.log(x), anoms))
    norms_log = list(map(lambda x: math.log(x), norms))
    
    data = np.append(anoms_log, norms_log)
    
    anoms_normalized = list(map(lambda x: (x-min(data))/(max(data)-min(data)), anoms_log))
    norms_normalized = list(map(lambda x: (x-min(data))/(max(data)-min(data)), norms_log))
    
    norm_end = time()
    logger.info("Done normalizing anomaly scores (%s seconds)" % (norm_end-norm_start))
    
    min_score = min(np.append(anoms_normalized, norms_normalized))
    max_score = max(np.append(anoms_normalized, norms_normalized))
    
    bins = np.linspace(min_score, max_score, num_bins+1)    # list of bin edges
    bins = bins[1:num_bins]
    bin_norms = np.digitize(norms_normalized, bins, right=True)    # list of bin number per score

    count_norms = np.zeros(num_bins)                    # counts of bin
    for i in bin_norms:
        count_norms[i] += 1
        
    bin_anoms = np.digitize(anoms_normalized, bins)
    count_anoms = np.zeros(num_bins)
    for i in bin_anoms:
        count_anoms[i] += 1
    
    bar_pos = np.arange(1, num_bins+1)    
    step_size = 5
    x_pos = np.arange(0, num_bins, step_size) 
    pad = (bins[1]-bins[0])/2
    lss2 = pad*2*step_size
    lss = bins[5]-bins[0]     # label step size
    x_label = list(map(round_scores, np.arange(min_score+pad, max_score, lss)))

    return count_norms, count_anoms, bins, bar_pos, x_pos, x_label 


def normalize_bins(norm_bins, anom_bins):
    norm_total, anom_total = 0, 0
    for i in norm_bins:
        norm_total += i
    for i in anom_bins:
        anom_total += i
    
    normalized_norm_bins = list(map(lambda x: round_scores(x/norm_total*100,2), norm_bins))
    normalized_anom_bins = list(map(lambda x: round_scores(x/anom_total*100,2), anom_bins))

    return normalized_norm_bins, normalized_anom_bins

def plot_images_hist(normal_scores, anomaly_scores, export_img, title: str = '', auc = None):
    """ Plot 2d histogram """
    
    logger = logging.getLogger()
    logger.info('Plotting histogram...')
    bar_width = 0.4
    num_bins = 21

    normal_scores_bin, anomaly_scores_bin, bins, bar_pos, x_pos, x_label = digitize_scores(normal_scores.tolist(), anomaly_scores.tolist(), num_bins)
    normalized_norm_sc_bin, normalized_anom_sc_bin = normalize_bins(normal_scores_bin, anomaly_scores_bin)
    fig, ax = plt.subplots()
    
    ax.bar(x=bar_pos-bar_width/2, height=normalized_norm_sc_bin, width=bar_width, label='Normal')
    ax.bar(x=bar_pos+bar_width/2, height=normalized_anom_sc_bin, width=bar_width, label='Crack')
    plt.ylabel('Frequency (%)')
        
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels(x_label, rotation=30)
    ax.set_xticks([0,22])
    ax.set_xticklabels([0,1])
    
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Anomaly score')
    plt.legend()
    
    if not (title == ''):
        if auc != None:
            plt.title(title + ' (AUC = ' + str(round(auc, 4)*100) + '%)')
        else:
            plt.title(title)
    
    img = Image.new(mode='L', size=(4,4))
    filename = export_img + '_' + str(datetime.now()) + '.png'
    img.save(filename, "PNG")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    logger.info('Plotted histogram.')
    