import random
import numpy as np
from random import shuffle
from numpy import matlib
from scipy.special import softmax


class DS3(object):

    def __init__(self, dist, reg):
        self.dist = dist
        self.reg = reg
        self.N = len(self.dist)

    def reg_loss(self, z, p):
        loss = 0
        for i in range(self.N):
            loss += np.linalg.norm(z[i], ord=p)
        return loss * self.reg

    def rep_loss(self, z):
        return (self.dist * z).sum()

    def loss(self, z, p):
        return self.reg_loss(z, p) + self.rep_loss(z)

    def loss_shot(self, shot, p):
        z = self.build_z_given_shot(self.dist, shot)
        return self.reg_loss(z, p) + self.rep_loss(z)

    @staticmethod
    def build_z_given_shot(dist, shot):
        z = np.zeros(dist.shape)
        z[np.asarray(shot)[dist[shot].argmin(axis=0)], np.arange(dist.shape[1])] = 1
        return z

    @staticmethod
    def build_shot_given_z(z):
        return z.sum(axis=1).nonzero()

    def ADMM(self, mu, epsilon, max_iter, p, previous_shot=[], previous_penalized=0, penalized_coef=None,
             cooperation=None):
        """
        :param mu:        penalty parameter.
        :param epsilon:   small value to check for convergence.
        :param max_iter:  maximum number of iterations to run this algorithm.
        :param p:         norm to be used.
        :param previous_shot: the rows that don't need to be penalized
        :param penalized_coef: the coef for rows to be penalized
        :returns: representative of the data, total number of representatives, and the objective function value.
        """
        if penalized_coef is None:
            penalized_coef = np.ones(len(self.dist))
        # initialize the ADMM class.
        G = ADMM(mu, epsilon, max_iter, self.reg)

        # run the ADMM algorithm.
        k, z_matrix = G.runADMM(self.dist, p, previous_shot, False, np.inf, np.inf, previous_penalized, penalized_coef,
                                cooperation)
        z_matrix        = np.sum(z_matrix,axis=1)
        learner_weight  = (z_matrix-np.min(z_matrix)+0.01)/(np.max(z_matrix)-np.min(z_matrix)+0.01)

        return learner_weight


class ADMM(object):
    """
        :param mu:        penalty parameter.
        :param epsilon:   small value to check for convergence.
        :param max_iter:  maximum number of iterations to run this algorithm.
        :param reg:       regularization parameter.
        """

    def __init__(self, mu, epsilon, max_iter, reg):
        self.mu       = mu
        self.epsilon  = epsilon
        self.max_iter = max_iter
        self.reg      = reg

    @staticmethod
    def solve_hard_constraint(C, tau):
        """
        This function solves the optimization program of
                    min ||C-Z||_F^2  s.t.  |Z|_{1,inf} <= tau
        :param C:      variable of the optimization.
        :param tau:    constraints
        :returns:      MxN coefficient matrix.
        """
        epsilon = 1e-4
        C_abs = np.sort(np.absolute(C), axis=1)
        M, N = C.shape
        tau_rows = np.zeros(M)
        grad_rows = C_abs.sum(axis=1)
        while tau > epsilon and np.sum(grad_rows) > epsilon:
            cnt_rows = (C_abs < epsilon).sum(axis=1)
            grad_rows = C_abs.sum(axis=1)
            next_steps = np.zeros(M)
            max_grad = grad_rows.max()
            max_grad_idx = [i for i in range(M) if grad_rows[i] > max_grad - epsilon]
            delta_grad = 1e4
            for r in max_grad_idx:
                if cnt_rows[r] < N:
                    delta_grad = min(delta_grad, C_abs[r, cnt_rows[r]] * (N - cnt_rows[r]))
            if len(max_grad_idx) < M:
                second_large_grad = np.max([grad_rows[i] for i in range(M) if i not in max_grad_idx])
                delta_grad = min(delta_grad, max_grad - second_large_grad)
            delta_tau = [delta_grad / (N - cnt_rows[r]) if cnt_rows[r] < N else 0 for r in max_grad_idx]
            delta_tau_sum = np.sum(delta_tau)
            if delta_tau_sum > tau:
                coef = tau / delta_tau_sum
                delta_tau = [x * coef for x in delta_tau]
                delta_tau_sum = tau

            tau -= delta_tau_sum
            for i, r in enumerate(max_grad_idx):
                tau_rows[r] += delta_tau[i]
                C_abs[r] -= delta_tau[i]
                C_abs[r][C_abs[r] < 0] = 0

        Z = np.zeros(C.shape)
        for i in range(M):
            Z[i] = np.clip(C[i], -tau_rows[i], tau_rows[i])
        return Z

    @staticmethod
    def shrinkL2Linf(y, t):
        """
        This function minimizes
                0.5*||b*x-y||_2^2 + t*||x||_inf, where b is a scalar.
        Note that it suffices to consider the minimization
                0.5*||x-y||_2^2 + t/b*||x||_inf
        the value of b can be assumed to be absorbed into t (= tau).
        The minimization proceeds by initializing x with y.  Let z be y re-ordered so that the abs(z) is in
        descending order. Then first solve
                min_{b>=abs(z2)} 0.5*(b-abs(z1))^2 + t*b
        if b* = abs(z2), then repeat with first and second largest z values;
                min_{b>=abs(z3)} 0.5*(b-abs(z1))^2+0.5*(b-abs(z2))^2 + t*b
        which by expanding the square is equivalent to
                min_{b>=abs(z3)} 0.5*(b-mean(abs(z1),abs(z2)))^2 + t*b
        and repeat this process if b*=abs(z3), etc.
        This reduces problem to finding a cut-off index, where all coordinates are shrunk up to and
        including that of the cut-off index.  The cut-off index is the smallest integer k such that
               1/k sum(abs(z(1)),...,abs(z(k))) - t/k <= abs(z(k+1))
        :param y:       variable of the above optimization .
        :param t:       regualrization for the above optimization
        :returns:       row of MxN coefficient matrix.
        """

        x = np.array(y, dtype=np.float32)
        o = np.argsort(-np.absolute(y))
        z = y[o]

        # find the cut-off index
        cs = np.divide(np.cumsum(np.absolute(z[0:len(z) - 1])), (np.arange(1, len(z))).T) - \
             np.divide(t, np.arange(1, len(z)))
        d = np.greater(cs, np.absolute(z[1:len(z)])).astype(int)
        if np.sum(d, axis=0) == 0:
            cut_index = len(y)
        else:
            cut_index = np.min(np.where(d == 1)[0]) + 1

        # shrink coordinates 0 to cut_index - 1
        zbar = np.mean(np.absolute(z[0:cut_index]), axis=0)

        if cut_index < len(y):
            x[o[0:cut_index]] = np.sign(z[0:cut_index]) * max(zbar - t / cut_index, np.absolute(z[cut_index]))
        else:
            x[o[0:cut_index]] = np.sign(z[0:cut_index]) * max(zbar - t / cut_index, 0)

        return x

    @staticmethod
    def solverLpshrink(C1, l, p):
        """
        This function solves the shrinkage/thresholding problem for different norms p in {2, inf}
        :param C1:      variable of the optimization.
        :param l:       regualrization for the above optimization
        :param p:       norm used in the optimization
        :param previous_shot: the rows that don't need to be penalized
        :returns:       MxN coefficient matrix.
        """
        if len(l) > 0:
            [D, N] = np.shape(C1)

            if p == np.inf:
                C2 = np.zeros((D, N), dtype=np.float32)
                for i in range(D):
                    # if i in previous_shot:
                    #    C2[i, :] = C1[i, :]
                    # else:
                    C2[i, :] = ADMM.shrinkL2Linf(C1[i, :].T, l[i]).T
            elif p == 2:
                # raise NotImplementedError
                r = np.maximum(np.sqrt(np.sum(np.power(C1, 2), axis=1, keepdims=True)) - l, 0)
                C2 = np.multiply(matlib.repmat(np.divide(r, (r + l)), 1, N), C1)

        return C2

    @staticmethod
    def solverLpshrinkCooperation(C1, l, p, cooperation):
        [D, N] = np.shape(C1)
        if p == np.inf:
            C2 = np.zeros((D, N), dtype=np.float32)
            for i in range(D):
                C2[i, :] = ADMM.shrinkL2Linf(C1[i, :].T, l[i]).T
            for _ in range(10):
                for i in np.random.permutation(D):
                    new_coef = l[i] - np.sum([cooperation[i, j] * C2[j, :].max() for j in range(D)])
                    C2[i, :] = ADMM.shrinkL2Linf(C1[i, :].T, new_coef).T
        elif p == 2:
            raise NotImplementedError
            r = np.maximum(np.sqrt(np.sum(np.power(C1, 2), axis=1, keepdims=True)) - l, 0)
            C2 = np.multiply(matlib.repmat(np.divide(r, (r + l)), 1, N), C1)
            for _ in range(10):
                new_coef = np.zeros(D)
                for i in np.random.permutation(D):
                    new_coef[i] = l[i] - np.sum([cooperation[i, j] * C2[j, :].max() for j in range(D)])
                r = np.maximum(np.sqrt(np.sum(np.power(C1, 2), axis=1, keepdims=True)) - new_coef, 0)
                C2 = np.multiply(matlib.repmat(np.divide(r, (r + new_coef)), 1, N), C1)
        return C2


    @staticmethod
    def solverBCLSclosedForm(U):
        """
        This function solves the optimization program of
                    min ||C-U||_F^2  s.t.  C >= 0, 1^t C = 1^t
        :param U:      variable of the optimization.
        :returns:      MxN coefficient matrix.
        """

        [m, N] = np.shape(U)

        # make every row in decreasing order.
        V = np.flip(np.sort(U, axis=0), axis=0)

        # list to keep the hold of valid indices which requires updates.
        activeSet = np.arange(0, N)
        theta = np.zeros(N)
        i = 0

        # loop till there are valid indices present to be updated or all rows are done.
        while len(activeSet) > 0 and i < m:
            j = i + 1

            # returns 1 if operation results in negative value, else 0.
            idx = np.where((V[i, activeSet] - ((np.sum(V[0:j, activeSet], axis=0) - 1) / j)) <= 0, 1, 0)

            # find indices where the above operation is negative.
            s = np.where(idx == 1)[0]

            if len(s):
                theta[activeSet[s]] = (np.sum(V[0:i, activeSet[s]], axis=0) - 1) / (j - 1)

            # delete the indices which were updated this iteration.
            activeSet = np.delete(activeSet, s)
            i = i + 1

        if len(activeSet) > 0:
            theta[activeSet] = (np.sum(V[0:m, activeSet], axis=0) - 1) / m

        C = np.maximum((U - matlib.repmat(theta, m, 1)), 0)

        return C

    @staticmethod
    def errorCoef(Z, C):
        """
        This function computes the maximum error between elements of two coefficient matrices
        :param Z:       MxN coefficient matrix.
        :param C:       MxN coefficient matrix
        :returns:       infinite norm error between vectorized C and Z.
        """

        err = np.sum(np.sum(np.absolute(Z - C), axis=0), axis=0) / (np.size(Z, axis=0) * np.size(Z, axis=1))

        return err

    def runADMM(self, dis_matrix, p, previous_shot, outlier, beta, tau, previous_penalized, penalized_coef,
                cooperation):
        """
        This function solves the proposed trace minimization regularized by row-sparsity norm using an ADMM framework
        To know more about this, please read :
        Dissimilarity-based Sparse Subset Selection
        by Ehsan Elhamifar, Guillermo Sapiro, and S. Shankar Sastry
        https://arxiv.org/pdf/1407.6810.pdf
        :param dis_matrix:      dissimilarity matrix.
        :param p:               norm of the mixed L1/Lp regularizer, {2,inf}
        :returns:               representative matrix for te dataset.
        """
        if outlier:
            [M, N] = np.shape(dis_matrix)
            _dis_matrix = np.array(dis_matrix)
            dis_matrix = np.zeros((M + 1, N))
            dis_matrix[:M, :] = _dis_matrix
            for j in range(N):
                dis_matrix[M, j] = beta * np.exp(-_dis_matrix[:, j].min() / tau)
            previous_shot.append(M)

        [M, N] = np.shape(dis_matrix)
        k = 1

        # calculate te centroid point of te dataset.
        C1 = np.zeros((np.shape(dis_matrix)))
        idx = np.argmin(np.sum(dis_matrix, axis=1))
        C1[idx, :] = 1

        # regularization coefficient matrix.
        Lambda = np.zeros((M, N))
        CFD = np.ones((M, 1)) * (self.reg / self.mu) * penalized_coef.reshape(-1, 1)
        for i in previous_shot:
            CFD[i] *= previous_penalized
            # CFD[i] = previous_penalized
        while True:

            # perform the iterative ADMM steps for two variables.
            if cooperation:
                Z = self.solverLpshrinkCooperation(C1 - np.divide((Lambda + dis_matrix), self.mu), CFD, p, cooperation)
            else:
                Z = self.solverLpshrink(C1 - np.divide((Lambda + dis_matrix), self.mu), CFD, p)
            C2 = self.solverBCLSclosedForm(Z + np.divide(Lambda, self.mu))
            Lambda = Lambda + np.multiply(self.mu, (Z - C2))

            # calculate the error from previous iteration.
            err1 = self.errorCoef(Z, C2)
            err2 = self.errorCoef(C1, C2)
            # if error is less then epsilon then return the current representative matrix.
            if k >= self.max_iter or (err1 <= self.epsilon and err2 <= self.epsilon):
                break
            else:
                k += 1

            C1 = C2
        Z = C2

        return k, Z