import os
import numpy as np
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import scipy.spatial
import scipy.sparse
import community
import networkx
import logging


# if syslog is True，append log to /var/log/syslog
def create_logger(app_name, logfilename=None, level=logging.INFO,
                  console=False, syslog=False):
    """ Build and return a custom logger. Accepts the application name,
    log filename, loglevel and console logging toggle and syslog toggle """

    log = logging.getLogger(app_name)
    log.setLevel(level)
    # Add file handler
    if logfilename is not None:
        log.addHandler(logging.FileHandler(logfilename))

    if syslog:
        log.addHandler(logging.handlers.SysLogHandler(address='/dev/log'))

    if console:
        log.addHandler(logging.StreamHandler())

    # Add formatter
    for handle in log.handlers:
        formatter = logging.Formatter('%(asctime)s : %(levelname)-8s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handle.setFormatter(formatter)
    return log




def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size, 'y_pred.size {} y_true.size {}'.format(y_pred.size, y_true.size)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class SingleCellData(object):

    def __init__(self, geoid):
        self.data_path = os.path.join('dataset', geoid)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError("don't find single cell {} data".format(geoid))

        self.cell_num = None
        self.gene_num = None
        self.label_num = None
        self.id2gene = None
        self.gene2id = None
        self.id2cell = None
        self.cell2id = None
        self.id2label = None
        self.label2id = None
        self.expression_matrix = None
        self.labels = None
        self.__load_data()

    def __load_data(self):
        # load all cell name and statistic the number of genes
        with open(os.path.join(self.data_path, 'data.txt')) as f:
            cell_line = f.readline().strip()
            all_cells = cell_line.split('\t')[1:]
            self.cell_num = len(all_cells)
            self.id2cell = all_cells
            self.cell2id = {c_name:i for i, c_name in enumerate(self.id2cell)}
            gene_num = 0
            for l in f:
                l = l.strip()
                if l:
                    gene_num += 1
            self.gene_num = gene_num
            self.expression_matrix = np.zeros((self.cell_num, self.gene_num))
        self.id2gene = []
        with open(os.path.join(self.data_path, 'data.txt')) as f:
            f.readline()
            for i in range(self.gene_num):
                l = f.readline().strip()
                items = l.split('\t')
                gene_name = items[0]
                self.id2gene.append(gene_name)
                assert len(items[1:]) == self.cell_num
                for j, v in enumerate(items[1:]):
                    v = float(v)
                    self.expression_matrix[j, i] = float(v)
        self.gene2id = {g_name:i for i, g_name in enumerate(self.id2gene)}
        self.expression_matrix = np.log(self.expression_matrix + 1)

        with open(os.path.join(self.data_path, 'label.txt')) as f:
            label_names = []
            for l in f:
                l = l.strip()
                if l:
                    label_names.append(l)
            assert len(label_names) == self.cell_num
            self.id2label = list(set(label_names))
            self.label2id = {l_name:i for i, l_name in enumerate(self.id2label)}
            self.label_num = len(self.id2label)
            self.labels = np.array([self.label2id[l_name] for l_name in label_names], dtype=np.int64)

    @staticmethod
    def get_shrunk_data_by_std(single_cell_dataset, gene_num):
        assert isinstance(single_cell_dataset, SingleCellData)
        if single_cell_dataset.gene_num <= gene_num:
            print('given number is less than data\'s gene num')
            return single_cell_dataset
        gene_std = np.std(single_cell_dataset.expression_matrix, axis=0)
        selected_gene_id_lst = np.argsort(-gene_std)[:gene_num]
        id2gene = [single_cell_dataset.id2gene[i] for i in selected_gene_id_lst]
        gene2id = {g_name: i for i, g_name in enumerate(id2gene)}
        expression_matrix = np.ascontiguousarray(single_cell_dataset.expression_matrix[:, selected_gene_id_lst])
        data_copy = copy.deepcopy(single_cell_dataset)
        data_copy.id2gene = id2gene
        data_copy.gene2id = gene2id
        data_copy.gene_num = len(id2gene)
        data_copy.expression_matrix = expression_matrix
        return data_copy


class FeatureTransform(object):

    def __init__(self):
        self.verbose = True
        self.random_state = 0
        self.raw_feat = None
        self.transformed_feat = None
        self.labels = None
        self.tsne_feat = None

    def fit_transform(self, feat):
        raise NotImplementedError('not implement')

    def set_random_state(self, random_state):
        self.random_state = random_state

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_feature(self, raw_feat):
        self.raw_feat = raw_feat
        self.transformed_feat = self.fit_transform(raw_feat)

    def set_labels(self, labels):
        self.labels = labels

    def dump_distance_matrix(self, dumpdir):
        distance = cal_distance_matrix(self.transformed_feat, self.labels)
        np.savetxt(os.path.join(dumpdir, 'distance_matrix.txt'), distance)

    def dump_results(self, dumpdir):
        # self.dump_transformed_feat(os.path.join(dumpdir, 't_feat.txt'))
        self.dump_tsne_feat(os.path.join(dumpdir, 'v_feat.txt'))
        # self.dump_distance_matrix(dumpdir)
        with open(os.path.join(dumpdir, 'labels.txt'),'w') as f:
            for l in self.labels:
                f.write("{}\n".format(int(l)))

    def dump_transformed_feat(self, filename):
        assert self.transformed_feat is not None
        np.savetxt(filename, self.transformed_feat)

    def dump_tsne_feat(self, filename):
        if self.tsne_feat is None:
            assert self.transformed_feat is not None
            tsne = TSNE(n_components=2, verbose=self.verbose, random_state=self.random_state)
            tsne_feat = tsne.fit_transform(self.transformed_feat)
            self.tsne_feat = tsne_feat
        np.savetxt(filename, self.tsne_feat)


class IdentityTransform(FeatureTransform):

    def __init__(self, n_components=-1):
        super().__init__()
        self.n_components = n_components

    def fit_transform(self, feat):
        return feat


class PCAFeatureTransform(FeatureTransform):

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=self.random_state)

    def fit_transform(self, feat):
        t_feat = self.pca.fit_transform(feat)
        return t_feat


class ICAFeatureTransform(FeatureTransform):

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.ica = FastICA(n_components=n_components, random_state=self.random_state)

    def fit_transform(self, feat):
        t_feat = self.ica.fit_transform(feat)
        return t_feat


class NMFFeatureTransform(FeatureTransform):

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.nmf = NMF(n_components=n_components, random_state=self.random_state, verbose=True)

    def fit_transform(self, feat):
        t_feat = self.nmf.fit_transform(feat)
        return t_feat


class ClusteringModel(object):

    def __init__(self):
        self.random_state = 0
        self.verbose = True
        self.raw_feat = None
        self.pred_labels = None
        self.labels = None

    def fit_predict(self, feat):
        raise NotImplementedError('not implement')

    def set_random_state(self, random_state):
        self.random_state = random_state

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_feature(self, raw_feat):
        self.raw_feat = raw_feat
        pred = self.fit_predict(self.raw_feat)
        self.pred_labels = pred

    def set_labels(self, labels):
        self.labels = labels

    def dump_distance_matrix(self, dumpdir):
        distance = cal_distance_matrix(self.raw_feat, self.pred_labels)
        np.savetxt(os.path.join(dumpdir, 'distance_matrix.txt'), distance)

    def dump_results(self, dumpdir):
        assert self.pred_labels is not None
        # self.dump_distance_matrix(dumpdir)
        # np.savetxt(os.path.join(dumpdir, 'pred.txt'), self.pred_labels)
        with open(os.path.join(dumpdir, 'pred.txt'), 'w') as f:
            for l in self.pred_labels:
                f.write("{}\n".format(int(l)))
        if self.labels is not None:
            # np.savetxt(os.path.join(dumpdir, 'labels.txt'), self.labels)
            with open(os.path.join(dumpdir, 'labels.txt'), 'w') as f:
                for l in self.labels:
                    f.write("{}\n".format(int(l)))
            acc = cluster_acc(self.labels, self.pred_labels)
            nmi = normalized_mutual_info_score(self.labels, self.pred_labels)
            ari = adjusted_mutual_info_score(self.labels, self.pred_labels)
            with open(os.path.join(dumpdir, 'scores.txt'), 'w') as f:
                f.write('acc: {} nmi: {} ari: {}\n'.format(acc, nmi, ari))


def cal_distance_matrix(latent, labels=None):
    if labels is None:
        order_latent = latent
    else:
        order_latent = np.vstack([x for _, x in sorted(zip(labels, latent), key=lambda pair: pair[0])])
    distance = scipy.spatial.distance_matrix(order_latent, order_latent)
    return distance


class KMeansModel(ClusteringModel):

    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state,
                             verbose=self.verbose,
                             n_init=5, n_jobs=5)

    def fit_predict(self, feat):
        pred = self.kmeans.fit_predict(feat)
        return pred


class HierarchicalModel(ClusteringModel):

    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.hierachical = AgglomerativeClustering(n_clusters=n_clusters)

    def fit_predict(self, feat):
        pred = self.hierachical.fit_predict(feat)
        return pred


class APModel(ClusteringModel):

    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        self.ap = AffinityPropagation(verbose=True)

    def fit_predict(self, feat):
        pred = self.ap.fit_predict(feat)
        return pred


class DensityPeakModel(ClusteringModel):

    def __init__(self, n_clusters=10):
        super().__init__()
        self.n_clusters= n_clusters

    def fit_predict(self, feat):
        import pydpc

        class MyDensityPeak(pydpc.Cluster):

            def __init__(self, feat, n_clusters):
                super().__init__(feat, fraction=0.02, autoplot=False)
                self.MY_n_clusters = n_clusters

            def _get_cluster_indices(self):
                data_size = self.density.shape[0]
                density_idx = np.argsort(-self.density)
                delta_idx = np.argsort(-self.delta)
                ideal_n_clusters = self.MY_n_clusters
                idx = ideal_n_clusters
                while True:
                    tmp_density_idx = density_idx[:idx]
                    tmp_delta_idx = delta_idx[:idx]
                    n = len(set(tmp_density_idx).intersection(set(tmp_delta_idx)))
                    if n >= ideal_n_clusters:
                        break
                    idx += 1
                    if idx >= data_size / 20 or idx >= 1.5 * ideal_n_clusters:
                        break
                min_density = self.density[density_idx[idx]]
                min_delta = self.delta[delta_idx[idx]]

                self.clusters = np.intersect1d(
                    np.where(self.density > min_density)[0],
                    np.where(self.delta > min_delta)[0], assume_unique=True).astype(np.intc)
                self.nclusters = self.clusters.shape[0]

        clu = MyDensityPeak(feat, n_clusters=self.n_clusters)
        clu.assign(20, 1.5)
        pred = clu.membership
        return pred


class LouvainModel(ClusteringModel):

    def __init__(self, n_clusters=0, n_neighbors=10):
        super().__init__()
        self.n_neighbors = n_neighbors

    def fit_predict(self, feat):
        nne = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nne.fit(feat)
        kmatrix = nne.kneighbors_graph(feat) - scipy.sparse.identity(feat.shape[0])
        G = networkx.from_scipy_sparse_matrix(kmatrix)
        partition = community.best_partition(G)
        pred = []
        for i in range(feat.shape[0]):
            pred.append(partition[i])
        return np.array(pred)




if __name__ == '__main__':
    # INFO:root:Addition of 10 and 20 produces 30
    logging.basicConfig(filename='log.txt',
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    def clustering_pipeline(sc_dataset, DR_methods, Cluster_methods, feature_dim_lst, dumpdir):
        # the list n_neighbors is used to lovain community detection algorithm
        n_neighbors_lst = [20, 50, 60, 70, 80, 100, 120, 150, 200, 500, 1000]
        assert isinstance(sc_dataset, SingleCellData)
        n_clusters = sc_dataset.label_num
        gene_num = sc_dataset.gene_num
        for dr_method in DR_methods.keys():
            for n_component in feature_dim_lst:
                if dr_method == 'identity':
                    dr_results_dir = os.path.join(dumpdir, dr_method)
                    logging.info('Begin to run {}'.format(dr_method))
                    n_component = sc_data.gene_num
                else:
                    dr_results_dir = os.path.join(dumpdir, '{}_{}'.format(dr_method, n_component))
                    # gene_num must be less than n_component
                    if gene_num < n_component:
                        continue
                    logging.info('Begin to run {}_{}'.format(dr_method, n_component))
                if not os.path.exists(dr_results_dir):
                    os.makedirs(dr_results_dir)
                dr_model = DR_methods[dr_method](n_components=n_component)
                dr_model.set_feature(sc_dataset.expression_matrix)
                dr_model.set_labels(sc_dataset.labels)
                logging.info('Dump tsne feat and distance matrix for {}_{}'.format(dr_method, n_component))
                dr_model.dump_results(dr_results_dir)
                for cluster_method in Cluster_methods.keys():
                    logging.info('Begin to run {} clustering for {}_{}'.format(cluster_method, dr_method, n_component))
                    for n_neighbor in n_neighbors_lst:
                        #
                        if cluster_method != 'lovain':
                            cluster_results_dir = os.path.join(dr_results_dir, cluster_method)
                        else:
                            cluster_results_dir = os.path.join(dr_results_dir, '{}_{}'.format(cluster_method, n_neighbor))
                        if not os.path.exists(cluster_results_dir):
                            os.makedirs(cluster_results_dir)
                        #
                        if cluster_method != 'lovain':
                            cluster_model = Cluster_methods[cluster_method](n_clusters=n_clusters)
                        else:
                            cluster_model = Cluster_methods[cluster_method](n_clusters=n_clusters, n_neighbors=n_neighbor)
                        cluster_model.set_feature(dr_model.transformed_feat)
                        cluster_model.set_labels(sc_dataset.labels)
                        logging.info('Dump {} clustering for {}_{}\' results'.format(cluster_method, dr_method, n_component))
                        cluster_model.dump_results(cluster_results_dir)
                        if cluster_method != 'lovain':
                            break
                if dr_method == 'identity':
                    break

    gse_series = 'GSE60361'
    result_dir = os.path.join('results', gse_series)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logging.info('Loading dataset {}'.format(gse_series))
    cortex_data = SingleCellData(gse_series)
    import json
    with open(os.path.join(result_dir, 'label2id.json'), 'w') as f:
        f.write(json.dumps(cortex_data.label2id))

    DR_methods = {
        'identity': IdentityTransform,
        'pca': PCAFeatureTransform,
        'ica': ICAFeatureTransform,
        'nmf': NMFFeatureTransform
    }
    Cluster_methods = {
        #'dp': DensityPeakModel,
        'lovain': LouvainModel,
        'kmeans': KMeansModel,
        'hierachical': HierarchicalModel,
    }
    gene_num_lst = [500, 1000, 2000, 3000, -1]
    feature_dim_lst = [20, 30, 50, 100, 200, 300, 400, 500, 1000]

    logging.info('Algorithm begin to run')
    import pymp
    with pymp.Parallel(len(gene_num_lst)) as p:
        for i in p.range(len(gene_num_lst)):
            gene_num = gene_num_lst[i]
            if gene_num == -1:
                sc_data = cortex_data
                gene_num = sc_data.gene_num
            else:
                logging.info('Getting a shrunk dataset with gene num {}'.format(gene_num))
                sc_data = SingleCellData.get_shrunk_data_by_std(cortex_data, gene_num=gene_num)
            dumpdir = os.path.join(result_dir, str(gene_num))
            if not os.path.exists(dumpdir):
                os.makedirs(dumpdir)
            clustering_pipeline(sc_data, DR_methods, Cluster_methods, feature_dim_lst, dumpdir)

