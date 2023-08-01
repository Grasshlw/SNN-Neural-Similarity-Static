import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold


class SimilarityMetric:
    def __init__(self, seed=2022):
        self.seed = seed
    
    def _pearson_correlation_coefficient(self, x, y, mode="normal"):
        if mode == "cross":
            x_mean = np.mean(x, axis=1, keepdims=True)
            y_mean = np.mean(y, axis=1, keepdims=True)

            x_center = x - x_mean
            y_center = y - y_mean

            x_diag = np.diagonal(np.dot(x_center, x_center.T)).reshape((-1, 1))
            y_diag = np.diagonal(np.dot(y_center, y_center.T)).reshape((1, -1))
            r = np.dot(x_center, y_center.T) / np.sqrt(np.tile(x_diag, (1, x.shape[0])) * np.tile(y_diag, (y.shape[0], 1)))
        elif mode == "parallel":
            x_mean = np.mean(x, axis=0, keepdims=True)
            y_mean = np.mean(y, axis=0, keepdims=True)

            x_center = x - x_mean
            y_center = y - y_mean
        
            r = np.sum(x_center * y_center, axis=0) / np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center, axis=0))
        elif mode == "normal":
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            x_center = x - x_mean
            y_center = y - y_mean
        
            r = np.sum(x_center * y_center) / np.sqrt(np.sum(x_center * x_center) * np.sum(y_center * y_center))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return r

    def _spearman_correlation_coefficient(self, x, y):
        x_rank = np.argsort(np.argsort(x)).astype("float64")
        y_rank = np.argsort(np.argsort(y)).astype("float64")
        n = x.shape[0]
        r = 1 - 6 * np.sum((x_rank - y_rank) ** 2) / (n ** 3 - n)

        return r

    def score(self, model_data, neural_data):
        pass


class CCAMetric(SimilarityMetric):
    def __init__(self, reduction="TSVD", dims=40, neural_reduction=True, seed=2022):
        super().__init__(seed)
        self.reduction = reduction
        self.dims = dims
        self.neural_reduction = neural_reduction
    
    def _cca(self, x, y):
        def matrix_sqrt(m):
            w, v = np.linalg.eigh(m)
            w_sqrt = np.sqrt(np.abs(w))
            return np.dot(v, np.dot(np.diag(w_sqrt), np.conj(v).T))

        x_num = x.shape[0]
        y_num = y.shape[0]

        covariance = np.cov(x, y)
        cov_xx = covariance[:x_num, :x_num]
        cov_xy = covariance[:x_num, x_num:]
        cov_yx = covariance[x_num:, :x_num]
        cov_yy = covariance[x_num:, x_num:]

        x_max = np.max(np.abs(cov_xx))
        y_max = np.max(np.abs(cov_yy))
        cov_xx /= x_max
        cov_yy /= y_max
        cov_xy /= np.sqrt(x_max * y_max)
        cov_yx /= np.sqrt(x_max * y_max)

        cov_xx_inv = np.linalg.pinv(cov_xx)
        cov_yy_inv = np.linalg.pinv(cov_yy)

        cov_xx_sqrt_inv = matrix_sqrt(cov_xx_inv)
        cov_yy_sqrt_inv = matrix_sqrt(cov_yy_inv)

        M = np.dot(cov_xx_sqrt_inv, np.dot(cov_xy, cov_yy_sqrt_inv))

        u, s, v = np.linalg.svd(M)
        s = np.abs(s)

        x_ = np.dot(np.dot(u.T, cov_xx_sqrt_inv), x)
        y_ = np.dot(np.dot(v, cov_yy_sqrt_inv), y)

        return s, x_, y_

    def score(self, model_data, neural_data):
        if self.reduction == "PCA":
            red_model = PCA(n_components=self.dims, random_state=self.seed)
            red_neural = PCA(n_components=self.dims, random_state=self.seed)
        elif self.reduction == "TSVD":
            red_model = TruncatedSVD(n_components=self.dims, random_state=self.seed)
            red_neural = TruncatedSVD(n_components=self.dims, random_state=self.seed)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")
        
        if self.dims < model_data.shape[1]:
            red_model.fit(model_data)
            model_lowd = red_model.transform(model_data)
        else:
            model_lowd = model_data.copy()
        if self.neural_reduction and self.dims < neural_data.shape[1]:
            red_neural.fit(neural_data)
            neural_lowd = red_neural.transform(neural_data)
        else:
            neural_lowd = neural_data.copy()
        
        model_lowd = model_lowd.transpose((1, 0))
        neural_lowd = neural_lowd.transpose((1, 0))

        s, _, _ = self._cca(model_lowd, neural_lowd)
        return np.mean(s)


class RSAMetric(SimilarityMetric): 
    def score(self, model_data, neural_data):
        num_classes = model_data.shape[0]

        model_RDM = 1 - self._pearson_correlation_coefficient(model_data, model_data, mode="cross")
        neural_RDM = 1 - self._pearson_correlation_coefficient(neural_data, neural_data, mode="cross")

        model_RDM = model_RDM[np.triu_indices(num_classes, 1)]
        neural_RDM = neural_RDM[np.triu_indices(num_classes, 1)]

        return self._spearman_correlation_coefficient(model_RDM, neural_RDM)


class RegMetric(SimilarityMetric):
    def __init__(self, reduction="TSVD", dims=40, splits=-1, seed=2022):
        super().__init__(seed)
        self.reduction = reduction
        self.dims = dims
        self.splits = splits
    
    def score(self, model_data, neural_data):
        num_classes = model_data.shape[0]

        if self.reduction == "PCA":
            red_model = PCA(n_components=self.dims, random_state=self.seed)
        elif self.reduction == "TSVD":
            red_model = TruncatedSVD(n_components=self.dims, random_state=self.seed)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")
        if self.dims < model_data.shape[1]:
            red_model.fit(model_data)
            model_lowd = red_model.transform(model_data)
        else:
            model_lowd = model_data.copy()
        
        neural_pred = np.zeros(neural_data.shape)
        if self.splits == -1:
            kf = KFold(n_splits=num_classes, shuffle=True, random_state=self.seed)
        else:
            kf = KFold(n_splits=self.splits, shuffle=True, random_state=self.seed)
        for train_index, test_index in kf.split(model_lowd):
            model_lowd_train = model_lowd[train_index]
            model_lowd_test = model_lowd[test_index]
            neural_train = neural_data[train_index]
            neural_test = neural_data[test_index]

            reg = Ridge(alpha=1.0)
            reg.fit(model_lowd_train, neural_train)
            neural_pred[test_index] = reg.predict(model_lowd_test)

        r = self._pearson_correlation_coefficient(neural_pred, neural_data, mode="parallel")
        return np.mean(r)


class CKAMetric(SimilarityMetric):
    def __init__(self, kernel="linear", seed=2022):
       super().__init__(seed)
       self.kernel = kernel

    def score(self, model_data, neural_data):
        num_classes = model_data.shape[0]
        
        def centering(K):
            I = np.eye(num_classes)
            H = I - np.ones((num_classes, num_classes)) / num_classes
            return np.dot(K, H)
        
        def linear_HSIC(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            return np.sum(centering(XX) * centering(YY).T)

        def rbf_HSIC(X, Y, sigma=None):
            def rbf(A, sigma=None):
                AA = np.dot(A, A.T)
                AA_diag = np.diag(AA)
                D_A = (AA_diag - AA).T - AA + AA_diag
                if sigma is None:
                    sigma = np.sqrt(np.median(D_A[D_A != 0]))
                return np.exp(-D_A / (2 * (sigma ** 2)))
            return np.sum(centering(rbf(X, sigma=sigma)) * centering(rbf(Y, sigma=sigma)).T)

        if self.kernel == "linear":
            return linear_HSIC(model_data, neural_data) / (np.sqrt(linear_HSIC(model_data, model_data)) * np.sqrt(linear_HSIC(neural_data, neural_data)))
        elif self.kernel == "rbf":
            return rbf_HSIC(model_data, neural_data) / (np.sqrt(rbf_HSIC(model_data, model_data)) * np.sqrt(rbf_HSIC(neural_data, neural_data)))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
