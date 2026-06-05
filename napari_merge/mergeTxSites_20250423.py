#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:13:07 2025

@author: rachel
"""

import pandas as pd
import bigfish.stack as stack
from skimage import io
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage import measure
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from cellpose.io import imread, save_to_png, masks_flows_to_seg, imsave
from cellpose.utils import remove_edge_masks
from skimage import io, data
import math
from magicgui import magicgui, magic_factory, widgets
import napari
import math
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog
# from skimage import label

def get_file_name(title, homedir='./'):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(initialdir = homedir, title=title)  # Open file dialog

    root.destroy()  # Close the tkinter window
    return file_path

def getNucleiCoordinates(label_img, shouldIplot=False):
    """Extract nuclei coordinates and related information from a binary mask.

    Parameters
    ----------
    maskPath : str
        Path to the binary mask image containing nuclei.
    shouldIplot : bool, optional
        Flag indicating whether to plot the nuclei and related information, default is False.

    Returns
    -------
    tuple
        A tuple containing the following lists:
        1. cropBoxCoordinates : list
            List of crop box coordinates for each detected nucleus.
        2. nucleiCentroids : list
            List of centroids for each detected nucleus.
        3. noNuclei : ndarray
            Array of unique nucleus labels.
        4. orientations : list
            List of orientation information for each detected nucleus.

    Notes
    -----
    This function reads a binary mask image, labels connected components, and extracts
    information such as crop box coordinates, centroids, nucleus labels, and orientations.
    If shouldIplot is set to True, it also plots the nuclei and related information.

    Example
    -------
    crop_boxes, centroids, nucleus_labels, orientations = getNucleiCoordinates('/path/to/mask/image.tif', shouldIplot=True)
    """
    # image = io.imread(maskPath)
    label_img = remove_edge_masks(label_img, change_index=False)
    regions = regionprops(label_img)
    noNuclei = np.unique(label_img)
    noNuclei = np.delete(noNuclei,0)
    if shouldIplot==True:
        fig, ax = plt.subplots()
        ax.imshow(label_img, cmap=plt.cm.gray)
    cropBoxCoordinates = []
    nucleiCentroids = []
    orientations = []
    kk=0
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        if shouldIplot==True:           
            ax.plot((x0, x1), (y0, y1), '-r', linewidth=.7)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=.7)
            ax.plot(x0, y0, '.g', markersize=15)  
        minr_, minc_, maxr_, maxc_ = props.bbox
        maxcc = np.max([abs(minr_-maxr_),abs(minc_-maxc_)])
        minr = minr_-0.1*maxcc
        minc = minc_-0.1*maxcc
        maxr = maxr_+0.1*maxcc
        maxc = maxc_+0.1*maxcc
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if shouldIplot==True:            
            ax.plot(bx, by, '-b', linewidth=.7)
            ax.text(x0,y0,noNuclei[kk], color='white')

        nucleiCentroids.append([y0,x0])
        cropBoxCoordinates.append([bx,by])
        orientations.append([x1,y1,x2,y2])
        kk=kk+1
    if shouldIplot==True: 
        ax.axis((0, 1024, 1024, 0))
    plt.show()
    return cropBoxCoordinates, nucleiCentroids, noNuclei, orientations


def mergeTxSites(proposalCluster):
    newCluster = np.zeros((1,5))
    newCluster[0,0:3]=proposalCluster[np.argmax(proposalCluster[:,3]), 0:3]
    newCluster[0,4]=proposalCluster[np.argmax(proposalCluster[:,3]), 4]
    newCluster[0,3]= np.sum(proposalCluster[:,3])
    return newCluster


def get_spots(spots):  
    points = np.array(spots['coordinates'])
    dfpoint = []
    for i in range(len(points)):
        pt = np.array([int(i.strip().replace('(','').replace(')','')) for i in points[i].split(',')])
        dfpoint.append(pt)
    points = np.vstack(dfpoint)
    return points


def get_centroid_of_points(spots_in_clusters):
    x = [p[0] for p in spots_in_clusters]
    y = [p[1] for p in spots_in_clusters]
    z = [p[2] for p in spots_in_clusters]
    centroid = np.round(sum(x) / len(spots_in_clusters)), np.round(sum(y) / len(spots_in_clusters)), np.round(sum(z) / len(spots_in_clusters))
    return np.array(centroid)
    

def get_clusters(spots):
    cluster_points = spots.loc[spots['cluster_id'] != -1]
    clusters_sorted = cluster_points.sort_values(by=['cluster_id'])
    clusters = clusters_sorted['cluster_id'].unique()
    cluster_array = []
    cell_labels = []
    for clust in clusters:
        clb = spots.loc[spots['cluster_id'] == clust]['cell_label'].unique()
        cell_labels.append([clb[0],clust])
        points = spots.loc[(spots['cluster_id'] == clust)]['coordinates'].to_list()
        if len(points)!=0:
            spots_in_clusters = []
            for i in range(len(points)):
                pt = np.array([int(i.strip().replace('(','').replace(')','')) for i in points[i].split(',')])
                spots_in_clusters.append(pt)
            centroid_cluster = get_centroid_of_points(spots_in_clusters)
            count_spots = len(spots_in_clusters)
            ct = np.hstack([np.array(centroid_cluster), count_spots, clust])
            cluster_array.append(ct)
    return np.array(cluster_array), cell_labels
        



data_file_name = Path('/home/rachel/Downloads/smifish_data_20250502/spot_detection/k11_basal_001.csv')
# data_file_name = Path(get_file_name('Find \'spots_extraction..\' file'))
file_name_name = data_file_name.name

exp_name = file_name_name.split('.')[0]
homePath = data_file_name.parent.absolute().parent

# nuc_label_file_name = Path(get_file_name('Find nuclei segmentation file', homedir=homePath))
# cell_label_file_name = Path(get_file_name('Find cell segmentation file', homedir=homePath))
# cell_label_file_name = Path(get_file_name('Find cell segmentation file', homedir=homePath))

# rna_mip_file_name = Path(get_file_name('Find mip file for RNA channel',homedir=homePath))
rna_mip_file_name = 'MAX_'+exp_name+'_CY3.tif'
nuc_label_file_name = os.path.join(homePath,'segmentation', exp_name+'_nucleus_segmentation.npy')
cell_label_file_name = os.path.join(homePath,'segmentation', exp_name+'_cytoplasm_segmentation.npy')
rna_mip_file_path = os.path.join(homePath, 'mips',rna_mip_file_name)
rna_mip = io.imread(rna_mip_file_path)
nuc_label = np.load(nuc_label_file_name)
cell_label = np.load(cell_label_file_name)



result_file = pd.read_csv(data_file_name, index_col=0, delimiter=',')

# rna_mip = io.imread(rna_mip_file_name)
# nuc_label = np.load(nuc_label_file_name)
# kernel_size_dilation = 2

# nuc_label_new = stack.dilation_filter(nuc_label.astype(bool), kernel_shape='disk', kernel_size=kernel_size_dilation)
# cell_label = np.load(cell_label_file_name)


clusters, matched_cell_labels = get_clusters(result_file)
spots = get_spots(result_file)

CLST = deepcopy(clusters)
MTCH = deepcopy(matched_cell_labels)
FLNM = file_name_name
FLPR = homePath

cropBoxCoordinates, nucleiCentroids, nucleiNumbers, _ = getNucleiCoordinates(cell_label, False)

polygon = []
for nuclei in range(len(cropBoxCoordinates)):
    bx = np.asarray(cropBoxCoordinates[nuclei][0])
    by = np.asarray(cropBoxCoordinates[nuclei][1])
    vertices = []
    for ii in range(len(bx)):
        vertices.append([by[ii], bx[ii]])
    polygon.append(vertices)

features = {
    'N': nucleiNumbers,
}
text = {
    'string': '{N:.1f}',
    'size': 12,
    'color': 'red',
    'translation': np.array([-20, 0]),
}

face_color_cycle = ['white']

viewer = napari.view_image(rna_mip, colormap='green')
# gfpchannel = viewer.add_image(gfp_mip, colormap='red')
labels_layer = viewer.add_labels(cell_label, name='cell',opacity=0.3)

labels_layer2 = viewer.add_labels(nuc_label, name='nuclei',opacity=0.3)
# shapes_layer = viewer.add_shapes(polygon, shape_type='polygon', edge_width=2,
#                           edge_color='white', face_color='#00000000', opacity=0.3)
points_layer = viewer.add_points(
    nucleiCentroids,
    features=features,
    text=text,
    size=2,
    edge_width=2,
    edge_width_is_relative=False,
    edge_color='N',
    edge_colormap='gray',
    face_color_cycle=face_color_cycle,
    name = 'nuclei Label'
)


bigfish_Spots = viewer.add_points(
    spots[:,1:3],
    face_color='#00000000',
    size=10,
    edge_width=0.4,
    edge_width_is_relative=False,
    edge_color='yellow',
    name = 'bigFish Detected Spots'
    )

features2 = {
    'T': clusters[:,3],
}
text2 = {
    'string': '{T:.1f}',
    'size': 8,
    'color': 'white',
    'translation': np.array([-20, 0]),
}

bigfish_clusters = viewer.add_points(
    clusters[:,1:3],
    features=features2,
    text=text2,
    face_color='#00000000',
    size=12,
    edge_width=1,
    edge_width_is_relative=False,
    edge_color='red',
    name = 'bigFish clusters'
    )

@magicgui(call_button='Merge Transcription site',main_window=False,
#           saveTableButton=dict(widget_type="PushButton", text="Save mrna data"),
#           loadMaskBtn=dict(widget_type="PushButton", text="Load Mask")
         )
def onAddTranscriptionSite():
    global CLST
    print(viewer.layers['bigFish clusters'].selected_data)
    clusters = deepcopy(CLST)
    proposalCluster = clusters[list(viewer.layers['bigFish clusters'].selected_data),:]
    newCluster = mergeTxSites(proposalCluster)
    clusters[list(viewer.layers['bigFish clusters'].selected_data)[0]] = newCluster
    indicestobe = list(viewer.layers['bigFish clusters'].selected_data)[1:]
    clusters = np.delete(clusters, indicestobe, 0)
    CLST = deepcopy(clusters)
    print('len cluster:',len(clusters))
    print('len cluster:',len(CLST))
    features2 = {
        'T': clusters[:,3],
    }
    text2 = {
        'string': '{T:.1f}',
        'size': 8,
        'color': 'white',
        'translation': np.array([-20, 0]),
    }
    if 'bigFish clusters' in viewer.layers:
        viewer.layers.remove('bigFish clusters')
        bigfish_clusters = viewer.add_points(
            clusters[:,1:3],
            features=features2,
            text=text2,
            face_color='#00000000',
            size=12,
            edge_width=1,
            edge_width_is_relative=False,
            edge_color='red',
            name = 'bigFish clusters'
            )
        
@magicgui(call_button='Remove Transcription site',main_window=False,
         )
def onRemoveTranscriptionSite():
    global CLST
    print(viewer.layers['bigFish clusters'].selected_data)
    clusters = deepcopy(CLST)
    indicestobe = list(viewer.layers['bigFish clusters'].selected_data)[:]
    clusters = np.delete(clusters, indicestobe, 0)
    CLST = deepcopy(clusters)
    features2 = {
        'T': clusters[:,3],
    }
    text2 = {
        'string': '{T:.1f}',
        'size': 8,
        'color': 'white',
        'translation': np.array([-20, 0]),
    }
    if 'bigFish clusters' in viewer.layers:
        viewer.layers.remove('bigFish clusters')
        bigfish_clusters = viewer.add_points(
            clusters[:,1:3],
            features=features2,
            text=text2,
            face_color='#00000000',
            size=12,
            edge_width=1,
            edge_width_is_relative=False,
            edge_color='red',
            name = 'bigFish clusters'
            )
@magicgui(call_button='go to cell',main_window=False,
         )        
def onGoToCell(cell_id = 0):
    if 'polygon' in viewer.layers:
        viewer.layers.remove('polygon')
    polygon = []
    idx = int(np.where(nucleiNumbers==cell_id)[0])
    print(idx)
    bx = np.asarray(cropBoxCoordinates[idx][0])
    by = np.asarray(cropBoxCoordinates[idx][1])
    vertices = []
    for ii in range(len(bx)):
        vertices.append([by[ii], bx[ii]])
    polygon.append(vertices)
    shapes_layer = viewer.add_shapes(polygon, shape_type='polygon', edge_width=2,
                          edge_color='white', face_color='#00000000', opacity=0.3)
    
    
@magicgui(call_button='Save Results in csv',main_window=False,
         )        
def saveResults():  
    global CLST, MTCH, FLNM, FLPR
    matched_cell_labels = deepcopy(MTCH)
    final_clusters = deepcopy(CLST)
    df = pd.DataFrame(final_clusters, columns=['z','y','x', 'rna', 'cluster_id'])
    df['cell_label'] = 0
    matched_cell_labels = np.array(matched_cell_labels)
    for i in range(len(df)):
        idx = df.iloc[i,4]
        idx_cell = np.where(idx==np.array(matched_cell_labels)[:,1])
        if len(idx_cell)!=0:
            df.iloc[i,5] = matched_cell_labels[idx_cell,0]
    df.to_csv(os.path.join(FLPR, FLNM.replace('.xlsx','_tx_results.csv')))
    
    
    
container = [onAddTranscriptionSite,
             onRemoveTranscriptionSite,
            onGoToCell,
            saveResults] 
    
viewer.window.add_dock_widget(container, name = 'Remove Ts')







