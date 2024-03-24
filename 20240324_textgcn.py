dataset = '20240316_textgcn_tag_elec_fold_10_data'
# dataset = 'R8'

datasets = ['20240316_textgcn_tag_elec_fold_1_data',
           '20240316_textgcn_tag_elec_fold_2_data',
           '20240316_textgcn_tag_elec_fold_3_data',
           '20240316_textgcn_tag_elec_fold_4_data',
           '20240316_textgcn_tag_elec_fold_5_data',
           '20240316_textgcn_tag_elec_fold_6_data',
           '20240316_textgcn_tag_elec_fold_7_data',
           '20240316_textgcn_tag_elec_fold_8_data',
           '20240316_textgcn_tag_elec_fold_9_data',
#            '20240316_textgcn_tag_elec_fold_10_data'
           ]

max_length = 128
batch_size = 64
m = 1
nb_epochs = 100
checkpoint_dir = None
gcn_layers = 2
gcn_model = 'gcn'
n_hidden = 200
heads = 8
dropout = 0.5
# gcn_lr = 1e-3 ##0.02
gcn_lr = 0.02
bert_lr = 2e-5

for dataset in datasets:


    import numpy as np
    import pickle as pkl
    import networkx as nx
    import scipy.sparse as sp
    # from scipy.sparse.linalg.eigen.arpack import eigsh ##this is changed
    from scipy.sparse.linalg import eigsh
    import sys
    import re


    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index


    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1    
        return np.array(mask, dtype = bool)


    def load_data(dataset_str):
        """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

        # training nodes are training docs, no initial features
        # print("x: ", x)
        # test nodes are training docs, no initial features
        # print("tx: ", tx)
        # both labeled and unlabeled training instances are training docs and words
        # print("allx: ", allx)
        # training labels are training doc labels
        # print("y: ", y)
        # test labels are test doc labels
        # print("ty: ", ty)
        # ally are labels for labels for allx, some will not have labels, i.e., all 0
        # print("ally: \n")
        # for i in ally:
        # if(sum(i) == 0):
        # print(i)
        # graph edge weight is the word co-occurence or doc word frequency
        # no need to build map, directly build csr_matrix
        # print('graph : ', graph)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # print(len(labels))

        idx_test = test_idx_range.tolist()
        # print(idx_test)
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


    def load_corpus(dataset_str):
        """
        Loads input corpus from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.train.index => the indices of training docs in original doc list.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """

        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, adj = tuple(objects)
        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
        print(len(labels))

        train_idx_orig = parse_index_file(
            "data/{}.train.index".format(dataset_str))
        train_size = len(train_idx_orig)

        val_size = train_size - x.shape[0]
        test_size = tx.shape[0]

        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + val_size)
        idx_test = range(allx.shape[0], allx.shape[0] + test_size)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


    def sparse_to_tuple(sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return sparse_to_tuple(features)


    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


    def preprocess_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return sparse_to_tuple(adj_normalized)


    def construct_feed_dict(features, support, labels, labels_mask, placeholders):
        """Construct feed dictionary."""
        feed_dict = dict()
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i]
                          for i in range(len(support))})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        return feed_dict


    def chebyshev_polynomials(adj, k):
        """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
        print("Calculating Chebyshev polynomials up to order {}...".format(k))

        adj_normalized = normalize_adj(adj)
        laplacian = sp.eye(adj.shape[0]) - adj_normalized
        largest_eigval, _ = eigsh(laplacian, 1, which='LM')
        scaled_laplacian = (
            2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

        t_k = list()
        t_k.append(sp.eye(adj.shape[0]))
        t_k.append(scaled_laplacian)

        def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
            s_lap = sp.csr_matrix(scaled_lap, copy=True)
            return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

        for i in range(2, k+1):
            t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

        return sparse_to_tuple(t_k)


    def loadWord2Vec(filename):
        """Read Word Vectors"""
        vocab = []
        embd = []
        word_vector_map = {}
        file = open(filename, 'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            if(len(row) > 2):
                vocab.append(row[0])
                vector = row[1:]
                length = len(vector)
                for i in range(length):
                    vector[i] = float(vector[i])
                embd.append(vector)
                word_vector_map[row[0]] = vector
        print('Loaded Word Vectors!')
        file.close()
        return vocab, embd, word_vector_map

    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    import os
    import random
    import numpy as np
    import pickle as pkl
    import networkx as nx
    import scipy.sparse as sp
    # from utils import loadWord2Vec, clean_str ##this is from the BertGCN folder
    from math import log
    from sklearn import svm
    from nltk.corpus import wordnet as wn
    from sklearn.feature_extraction.text import TfidfVectorizer
    import sys
    from scipy.spatial.distance import cosine
    from tqdm import tqdm

    word_embeddings_dim = 300
    word_vector_map = {}

    # shulffing
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()
    # print(doc_train_list)
    # print(doc_test_list)

    doc_content_list = []
    f = open('data/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    # print(doc_content_list)

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    print(train_ids)
    random.shuffle(train_ids)

    # partial labeled data
    #train_ids = train_ids[:int(0.2 * len(train_ids))]

    train_ids_str = '\n'.join(str(index) for index in train_ids)
    f = open('data/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    print(test_ids)
    random.shuffle(test_ids)

    test_ids_str = '\n'.join(str(index) for index in test_ids)
    f = open('data/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()

    ids = train_ids + test_ids
    print(ids)
    print(len(ids))

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('data/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()

    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    word_doc_list = {}

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    # label list
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    f = open('data/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()

    # x: feature vectors of training docs, no initial features
    # slect 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()

    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    print(y)

    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)
    print(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words

    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))

    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))


    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    '''
    Doc word heterogeneous graph
    '''

    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    # dump objects
    f = open("data/ind.{}.x".format(dataset), 'wb')
    pkl.dump(x, f)
    f.close()

    f = open("data/ind.{}.y".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()

    f = open("data/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()

    f = open("data/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()

    f = open("data/ind.{}.allx".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open("data/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()

    import torch as th
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F
    # from utils import *
    import dgl
    import torch.utils.data as Data
    from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
    from ignite.metrics import Accuracy, Loss, recall
    from sklearn.metrics import accuracy_score
    import numpy as np
    import os
    import shutil
    import argparse
    import sys
    import logging
    from datetime import datetime
    from torch.optim import lr_scheduler
    from model import BertGCN, BertGAT, GCN

    if checkpoint_dir is None:
        ckpt_dir = './checkpoint/{}_{}'.format(gcn_model, dataset)
    else:
        ckpt_dir = checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(os.path.basename('train_bert_gcn.py'), ckpt_dir)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logger = logging.getLogger('training logger')
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    cpu = th.device('cpu')
    gpu = th.device('cuda:0')

    # logger.info('arguments:')
    # logger.info(str(args))
    logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

    # Data Preprocess
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
    '''
    adj: n*n sparse adjacency matrix
    y_train, y_val, y_test: n*c matrices 
    train_mask, val_mask, test_mask: n-d bool array
    '''

    # compute number of real train/val/test/word nodes and number of classes
    nb_node = features.shape[0]
    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_word = nb_node - nb_train - nb_val - nb_test
    nb_class = y_train.shape[1]

    # instantiate model according to class number
    model = GCN(in_feats=nb_node,
                n_hidden=n_hidden,
                n_classes=nb_class,
                n_layers=gcn_layers-1,
                activation=F.elu,
                dropout=dropout )

    # transform one-hot label to class ID for pytorch computation
    y = y_train + y_test + y_val
    y_train = y_train.argmax(axis=1)
    y = y.argmax(axis=1)

    # document mask used for update feature
    doc_mask  = train_mask + val_mask + test_mask

    # build DGL Graph
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
    g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
        th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
    g.ndata['label_train'] = th.LongTensor(y_train)
    # g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))
    g.ndata['cls_feats'] = th.eye((nb_node)) ##for textGCN since we only use identity matrix as the input feature matrix


    logger.info('graph information:')
    logger.info(str(g))

    # create index loader
    train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
    val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
    test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
    idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

    from transformers import AdamW, get_linear_schedule_with_warmup

    # Training
    def update_feature():
        global model, g, doc_mask
        # no gradient needed, uses a large batchsize to speed up the process
        dataloader = Data.DataLoader(
            Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
            batch_size=1024
        )
        with th.no_grad():
            model = model.to(gpu)
            model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(gpu) for x in batch]
                output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
                cls_list.append(output.cpu())
            cls_feat = th.cat(cls_list, axis=0)
        g = g.to(cpu)
        g.ndata['cls_feats'][doc_mask] = cls_feat
        return g


    optimizer = th.optim.Adam([
            {'params': model.parameters(), 'lr': gcn_lr},
        ], lr= 0.02 # 1e-3 
    )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


    # Total number of training steps
    # total_steps = len(train_dataloader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0, # Default value
    #                                             num_training_steps=510)

    def train_step(engine, batch):
        global model, g, optimizer
        model.train()
        model = model.to(gpu)
        g = g.to(gpu)
        optimizer.zero_grad()
        (idx, ) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        train_mask = g.ndata['train'][idx].type(th.BoolTensor)
        gcn_logit = model(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        y_pred = th.nn.Softmax(dim=1)(gcn_logit)[train_mask]
        y_true = g.ndata['label_train'][idx][train_mask]
        loss = F.nll_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        g.ndata['cls_feats'].detach_()
        train_loss = loss.item()
        with th.no_grad():
            if train_mask.sum() > 0:
                y_true = y_true.detach().cpu()
                y_pred = y_pred.argmax(axis=1).detach().cpu()
                train_acc = accuracy_score(y_true, y_pred)
            else:
                train_acc = 1
        return train_loss, train_acc


    trainer = Engine(train_step)


    @trainer.on(Events.EPOCH_COMPLETED)
    def reset_graph(trainer):
        scheduler.step()
    #     update_feature() ##we turn this off if we want to use textGCN only
        th.cuda.empty_cache()


    def test_step(engine, batch):
        global model, g
        with th.no_grad():
            model.eval()
            model = model.to(gpu)
            g = g.to(gpu)
            (idx, ) = [x.to(gpu) for x in batch]
            gcn_logit = model(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
            y_pred = th.nn.Softmax(dim=1)(gcn_logit)
            y_true = g.ndata['label'][idx]
            return y_pred, y_true

    from ignite.engine import Engine, Events
    from ignite.handlers import EarlyStopping


    evaluator = Engine(test_step)

    metrics={
        'acc': Accuracy(),
        'nll': Loss(th.nn.NLLLoss()),
        'recall': recall.Recall(average = 'macro')
    }
    for n, f in metrics.items():
        f.attach(evaluator, n)

    # def score_function(engine):
    #     val_loss = engine.state.metrics['nll']
    #     return -val_loss

    # handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    # # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    # evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    # # https://github.com/pytorch/ignite/issues/445

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(idx_loader_train)
        metrics = evaluator.state.metrics
    #     train_acc, train_nll = metrics["acc"], metrics["nll"]
        train_recall, train_nll = metrics["recall"], metrics["nll"]
        evaluator.run(idx_loader_val)
        metrics = evaluator.state.metrics
    #     val_acc, val_nll = metrics["acc"], metrics["nll"]
        val_recall, val_nll = metrics["recall"], metrics["nll"]
        evaluator.run(idx_loader_test)
        metrics = evaluator.state.metrics
    #     test_acc, test_nll = metrics["acc"], metrics["nll"]
        test_recall, test_nll = metrics["recall"], metrics["nll"]
        logger.info(
            "Epoch: {}  Train recall: {:.4f} loss: {:.4f}  Val recall: {:.4f} loss: {:.4f}  Test recall: {:.4f} loss: {:.4f}"
            .format(trainer.state.epoch, train_recall, train_nll, val_recall, val_nll, test_recall, test_nll)
        )
        if val_recall > log_training_results.best_val_recall:
            logger.info("New checkpoint")
            th.save(
                {
                    'gcn': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': trainer.state.epoch,
                },
                os.path.join(
                    ckpt_dir, 'checkpoint.pth'
                )
            )
            log_training_results.best_val_recall = val_recall


    log_training_results.best_val_acc = 0
    log_training_results.best_val_loss = 0
    log_training_results.best_val_recall = 0
    # g = update_feature() ##we turn this off if we want to use textGCN only
    trainer.run(idx_loader, max_epochs=nb_epochs)

    import pandas as pd 

    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)


        idx_list = []
        for i in idx_loader_test:
            idx_list.extend(list(i[0].numpy()))
        idx = th.Tensor(idx_list)
        idx = idx.to(th.long).to(gpu)

        gcn_logit = model(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        y_pred = th.nn.Softmax(dim=1)(gcn_logit)
        y_true = g.ndata['label'][idx]

    y_pred_label_list = []
    for i in y_pred.detach().cpu().numpy():
        y_pred_label = np.argmax(i)
        y_pred_label_list.append(y_pred_label)

    y_true_label_list = list(y_true.detach().cpu().numpy())

    from sklearn.metrics import classification_report
    report = classification_report(y_true_label_list, y_pred_label_list, digits = 6,output_dict=True)

    # Convert the report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    report_df

    report_df.to_csv(f'{dataset}.csv', index=True)