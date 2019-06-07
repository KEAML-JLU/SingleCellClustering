import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import logging
import os
import json


def visualize_distance(distance_matrix_filename, label_filename, save_name):
    distance = np.loadtxt(distance_matrix_filename)
    labels = np.loadtxt(label_filename, dtype=np.int64)
    order_label = np.vstack([y for y in sorted(labels)])

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 70], height_ratios=[1, 70])
    gs.update(wspace=0.05, hspace=0.05)
    ax0 = plt.subplot(gs[1])
    # ax0.imshow(order_label.T, cmap='Dark2', interpolation='none', aspect=100)
    ax0.axis('off')
    ax1 = plt.subplot(gs[3], sharex=ax0)
    ax1.imshow(distance, cmap='hot', interpolation='none')
    ax1.axis('off')
    ax2 = plt.subplot(gs[2], sharey=ax1)
    # ax2.imshow(order_label, cmap='Dark2', interpolation='none', aspect=1/100.)
    ax2.axis('off')
    #plt.tight_layout()
    if save_name:
        fig.savefig(save_name)
    plt.close(fig)


def plot_tsne_feat(feat_filename, label_filename, save_name, label2id=None):
    def build_colors(label_num):
        fixed_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        cnames = {'brown': '#A52A2A','red': '#FF0000','green': '#008000','blue': '#0000FF','black': '#000000','yellow': '#FFFF00','blueviolet': '#8A2BE2','darkcyan': '#008B8B','orangered': '#FF4500','olive': '#808000','peru': '#CD853F','mediumpurple': '#9370DB','darkolivegreen': '#556B2F','cornflowerblue': '#6495ED','crimson': '#DC143C','hotpink': '#FF69B4','darkslateblue': '#483D8B','magenta': '#FF00FF','seagreen': '#2E8B57','greenyellow': '#ADFF2F'}
        #cnames = {'aliceblue': '#F0F8FF','aqua': '#00FFFF','aquamarine': '#7FFFD4','azure': '#F0FFFF','beige': '#F5F5DC','bisque': '#FFE4C4','black': '#000000','blanchedalmond': '#FFEBCD','blue': '#0000FF','blueviolet': '#8A2BE2','brown': '#A52A2A','burlywood': '#DEB887','cadetblue': '#5F9EA0','chartreuse': '#7FFF00','chocolate': '#D2691E','coral': '#FF7F50','cornflowerblue': '#6495ED','cornsilk': '#FFF8DC','crimson': '#DC143C','cyan': '#00FFFF','darkblue': '#00008B','darkcyan': '#008B8B','darkgoldenrod': '#B8860B','darkgreen': '#006400','darkkhaki': '#BDB76B','darkmagenta': '#8B008B','darkolivegreen': '#556B2F','darkorange': '#FF8C00','darkorchid': '#9932CC','darkred': '#8B0000','darksalmon': '#E9967A','darkseagreen': '#8FBC8F','darkslateblue': '#483D8B','darkturquoise': '#00CED1','darkviolet': '#9400D3','deeppink': '#FF1493','deepskyblue': '#00BFFF','dodgerblue': '#1E90FF','firebrick': '#B22222','forestgreen': '#228B22','fuchsia': '#FF00FF','gainsboro': '#DCDCDC','gold': '#FFD700','goldenrod': '#DAA520','green': '#008000','greenyellow': '#ADFF2F','honeydew': '#F0FFF0','hotpink': '#FF69B4','indianred': '#CD5C5C','indigo': '#4B0082','ivory': '#FFFFF0','khaki': '#F0E68C','lavender': '#E6E6FA','lavenderblush': '#FFF0F5','lawngreen': '#7CFC00','lemonchiffon': '#FFFACD','lightblue': '#ADD8E6','lightcoral': '#F08080','lightcyan': '#E0FFFF','lightgoldenrodyellow': '#FAFAD2','lightgreen': '#90EE90','lightpink': '#FFB6C1','lightsalmon': '#FFA07A','lightseagreen': '#20B2AA','lightskyblue': '#87CEFA','lightsteelblue': '#B0C4DE','lightyellow': '#FFFFE0','lime': '#00FF00','limegreen': '#32CD32','linen': '#FAF0E6','magenta': '#FF00FF','maroon': '#800000','mediumaquamarine': '#66CDAA','mediumblue': '#0000CD','mediumorchid': '#BA55D3','mediumpurple': '#9370DB','mediumseagreen': '#3CB371','mediumslateblue': '#7B68EE','mediumspringgreen': '#00FA9A','mediumturquoise': '#48D1CC','mediumvioletred': '#C71585','midnightblue': '#191970','mintcream': '#F5FFFA','mistyrose': '#FFE4E1','moccasin': '#FFE4B5','navy': '#000080','oldlace': '#FDF5E6','olive': '#808000','olivedrab': '#6B8E23','orange': '#FFA500','orangered': '#FF4500','orchid': '#DA70D6','palegoldenrod': '#EEE8AA','palegreen': '#98FB98','paleturquoise': '#AFEEEE','palevioletred': '#DB7093','papayawhip': '#FFEFD5','peachpuff': '#FFDAB9','peru': '#CD853F','pink': '#FFC0CB','plum': '#DDA0DD','powderblue': '#B0E0E6','purple': '#800080','red': '#FF0000','rosybrown': '#BC8F8F','royalblue': '#4169E1','saddlebrown': '#8B4513','salmon': '#FA8072','sandybrown': '#FAA460','seagreen': '#2E8B57','seashell': '#FFF5EE','sienna': '#A0522D','silver': '#C0C0C0','skyblue': '#87CEEB','slateblue': '#6A5ACD','snow': '#FFFAFA','springgreen': '#00FF7F','steelblue': '#4682B4','tan': '#D2B48C','teal': '#008080','thistle': '#D8BFD8','tomato': '#FF6347','turquoise': '#40E0D0','violet': '#EE82EE','wheat': '#F5DEB3','yellow': '#FFFF00','yellowgreen': '#9ACD32'}
        if label_num <= 7:
            return fixed_colors[:label_num]
        # assert label_num <= len(cnames)
        if label_num <= len(cnames):
            #color_lists = np.random.choice(list(cnames.keys()), label_num, replace=False)
            color_lists = list(cnames.keys())[:label_num]
        else:
            keys = list(cnames.keys())
            color_lists = (keys * (int(label_num / len(keys)) + 1))[:label_num]
        colors = [cnames[c] for c in color_lists]
        return colors

    feat_tsne = np.loadtxt(feat_filename)
    labels = np.loadtxt(label_filename, dtype=np.int64)
    if label2id is None:
        label2id = {i:i for i in set(labels)}
    colors = build_colors(len(label2id))
    c = [colors[i] for i in labels]
    tmp = [str(l) for l,i in label2id.items()]
    tmp_c = [colors[i] for l, i in label2id.items()]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.scatter(feat_tsne[:,0], feat_tsne[:,1], c=c, alpha=0.6, s=10)
    # pathes = [mpatches.Patch(color=color, label=l) for color, l in zip(tmp_c, tmp)]
    pathes = [mpatches.Patch(color=color) for color in tmp_c]
    fig.legend(loc=1, handles=pathes, labels=tmp)
    fig.savefig(save_name)
    plt.close(fig)


def plot_pipeline(run_dir, label2id):
    true_label_filename = os.path.join(run_dir, 'labels.txt')
    tsne_feat_filename = os.path.join(run_dir, 'v_feat.txt')
    if False:
        logging.info('visualize distance matrix in {}'.format(run_dir))
        visualize_distance(os.path.join(run_dir, 'distance_matrix.txt'),
                           true_label_filename,
                           os.path.join(run_dir, 'distance.png'))
    logging.info('visualize tsne feat with true labels in {}'.format(run_dir))
    plot_tsne_feat(tsne_feat_filename,
                   true_label_filename,
                   os.path.join(run_dir, 'tsne_feat.png'),
                   label2id)

    all_dir = list(os.listdir(run_dir))
    all_dir = [os.path.join(run_dir, d) for d in all_dir if os.path.isdir(os.path.join(run_dir, d))]
    import pymp
    with pymp.Parallel(len(all_dir)) as p:
        for i in p.range(len(all_dir)):
            cur_dir = all_dir[i]
            if False:
                logging.info('visualize distance matrix in {}'.format(cur_dir))
                visualize_distance(os.path.join(cur_dir, 'distance_matrix.txt'),
                                   os.path.join(cur_dir, 'labels.txt'),
                                   os.path.join(cur_dir, 'distance.png'))
            logging.info('visualize tsne feat with pred labels in {}'.format(cur_dir))
            plot_tsne_feat(tsne_feat_filename,
                           os.path.join(cur_dir, 'pred.txt'),
                           os.path.join(cur_dir, 'tsne_feat.png'))


if __name__ == '__main__':
    # INFO:root:Addition of 10 and 20 produces 30
    logging.basicConfig(filename='log_plot.txt',
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    # gse_series = 'GSE60361'
    gse_series = 'GSE71585'
    result_dir = os.path.join('results', gse_series)
    with open(os.path.join(result_dir, 'label2id.json')) as f:
        label2id = json.load(f)

    gene_num_dirs = list(os.listdir(result_dir))
    gene_num_dirs = [os.path.join(result_dir, d) for d in gene_num_dirs if os.path.isdir(os.path.join(result_dir, d))]
    for shrunk_dir in gene_num_dirs:
        sub_dirs = list(os.listdir(shrunk_dir))
        sub_dirs = [os.path.join(shrunk_dir, d) for d in sub_dirs if os.path.isdir(os.path.join(shrunk_dir, d))]
        for sub_dir in sub_dirs:
            plot_pipeline(sub_dir, label2id)


    


