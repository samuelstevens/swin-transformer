import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from PIL import Image

import graphviz

def create_hierarchical_tree_vis(features, hierarchy_labels, reduction_method="tsne", hierarchy_label_map=None, top_k=6, output="", rerun_reduction=False, verbose=False):
    assert reduction_method in ["tsne", "pca"], f"{reduction_method} not supported."
    
    def log(x):
        if verbose:
            print(x)

    features = features[:1000]
    hierarchy_labels = hierarchy_labels[:1000]

    lvls = len(hierarchy_labels[0])

    log(f"Running {reduction_method} on features: {features.shape}. This may take some time.")
    
    if reduction_method == "tsne":
        reduce = TSNE(2)
    elif reduction_method == "pca":
        reduce = PCA(2)

    if not rerun_reduction:
        features = reduce.fit_transform(features)

    data_queue = [(features, hierarchy_labels, 0)]
    largest_name = None
    graph_data = []
    while len(data_queue) > 0:
        past_largest_name = largest_name

        feats, lbls, lvl = data_queue.pop(0)
        if rerun_reduction:
            reduced_feats = reduce.fit_transform(feats)
        else:
            reduced_feats = feats
        if hierarchy_label_map is not None:
            lvl_lbl_map = hierarchy_label_map[lvl]
        
        log(f"Plotting Level {lvl}")

        lbl_lengths = []
        for lbl in set(lbls[:, lvl]):
            idx = lbls[:, lvl] == lbl
            lbl_lengths.append([lbl, len(reduced_feats[idx])])
        lbl_lengths = sorted(lbl_lengths, key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(20, 6))
        plt.axis('off')
        most_feats = None
        most_lbls = None
        highest_num = 0
        for lbl in set(lbls[:, lvl]):
            if top_k > 0:
                if lbl not in np.array(lbl_lengths)[:top_k, 0]: continue
            idx = lbls[:, lvl] == lbl
            feat = reduced_feats[idx]
            if hierarchy_label_map is not None:
                name = lvl_lbl_map[str(lbl)].split("_")[-1]
            else:
                name = f"{lbl}"
            plt.scatter(feat[:, 0], feat[:, 1], label=name)
            if len(feat) > highest_num:
                highest_num = len(feat)
                if rerun_reduction:
                    most_feats = feats[idx]
                else:
                    most_feats = feat
                most_lbls = lbls[idx]
                largest_name = lbl
                if hierarchy_label_map is not None:
                    largest_name = lvl_lbl_map[lbl]

        # Add to queue
        if (lvl+1) < lvls:
            data_queue.append((most_feats, most_lbls, lvl+1))

        plt.legend()
        if lvl > 0:
            save_path = os.path.join(output, f"depth_{lvl}_{past_largest_name}.png")
        else:
            save_path = os.path.join(output, f"depth_{lvl}.png")

        plt.savefig(save_path)
        graph_data.append((save_path, lvl))
        plt.close()

    g = graphviz.Digraph('Hierarchy', filename=os.path.join(output, 'heirarchy_image.gv'))
    edges = []

    for path, lvl in graph_data:
        if lvl > 0:
            edges.append([str(lvl-1), str(lvl)])
        #g.node(dp['name'], image=dp['path'], shape='rectangle', scale='false', fontsize='0', imagescale='true', fixedsize='true', height='1.5', width='3')
        g.node(str(lvl), image=path, shape='rectangle', fontsize='0')
    
    for edge in edges:
        g.edge(edge[0], edge[1])
    
    g.render(os.path.join(output, 'heirarchy_image'), format="png", view=False)

if __name__ == "__main__":
    import numpy as np
    features = np.load(f'/local/scratch/carlyn.1/swin_inat_results/val_features.npz')['features']
    labels = np.load(f'/local/scratch/carlyn.1/swin_inat_results/val_labels.npz')['labels']

    create_hierarchical_tree_vis(features, labels, reduction_method="tsne", output="../tmp/", verbose=True)
        
