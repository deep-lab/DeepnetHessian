import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from numpy import linalg as LA
from torch.autograd import Variable

class Hessian:
    def __init__(self,
                 crit=None,
                 loader=None,
                 device=None,
                 model=None,
                 num_classes=None,
                 hessian_type=None,
                 double=False,
                 spectrum_margin=None,
                 init_poly_deg=None,
                 poly_deg=None,
                 poly_points=None,
                 SSI_iters=None,
                 class_list=None,
                 vecs=[],
                 vals=[],
                 ):
        
        self.crit                  = crit
        self.loader                = loader
        self.device                = device
        self.model                 = model
        self.num_classes           = num_classes
        self.hessian_type          = hessian_type
        self.double                = double
        self.spectrum_margin       = spectrum_margin
        self.init_poly_deg         = init_poly_deg
        self.poly_deg              = poly_deg
        self.poly_points           = poly_points
        self.SSI_iters             = SSI_iters
        self.class_list            = class_list
        self.vecs                  = vecs
        self.vals                  = vals
        
        for i in range(len(self.vecs)):
            self.vecs[i] = self.my_device(self.vecs[i])
            
        f = getattr(nn, self.crit)
        self.criterion = f(reduction='sum')
        
    
    # computes matrix vector multiplication
    # where the matrix is either the Hessian, G or H
    def Hv(self, v):
        Hg = self.my_zero()
        counter = 0
        
        for iter, batch in enumerate(self.loader):
            
            input, target = batch[0], batch[1]
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            input = Variable(input)
            target = Variable(target)
            
            if self.double:
                input = input.double()
                
            f = self.model(input)
            
            loss = self.criterion(f, target)
            
            if self.hessian_type == 'G':
                z = torch.randn(f.shape)
                
                if self.double:
                    z = z.double()
                
                z = z.to(self.device)
                    
                z = Variable(z, requires_grad=True)
                
                # z^T (d f / d theta)
                zT_df_dtheta = torch.autograd.grad(f,
                                                   self.model.parameters(),
                                                   z,
                                                   create_graph=True)
                
                # v^T (z^T (d f / d theta)) / dz
                # (d f / d theta) v
                df_dtheta_v = torch.autograd.grad(zT_df_dtheta,
                                                  z,
                                                  v)
                
                dloss_df = torch.autograd.grad(loss,
                                               f,
                                               create_graph=True)
                
                d2loss_df2_df_dtheta_v = torch.autograd.grad(dloss_df,
                                                             f,
                                                             grad_outputs=df_dtheta_v)
                
                Hg_ = torch.autograd.grad(f,
                                          self.model.parameters(),
                                          grad_outputs=d2loss_df2_df_dtheta_v)
            elif self.hessian_type == 'H':
                dloss_df = torch.autograd.grad(loss,
                                               f)
                
                df_dtheta = torch.autograd.grad(f,
                                                self.model.parameters(),
                                                grad_outputs=dloss_df,
                                                create_graph=True)
                
                df_dtheta[-1].requires_grad = True
                
                Hg_ = torch.autograd.grad(df_dtheta,
                                          self.model.parameters(),
                                          v,
                                          allow_unused=True)
                
                zr = torch.zeros(df_dtheta[-1].shape)
                
                zr = zr.to(self.device)
                
                Hg_ = Hg_[:-1] + (zr,)
            elif self.hessian_type == 'Hessian':
                grad = torch.autograd.grad(loss,
                                           self.model.parameters(),
                                           create_graph=True)
                
                Hg_ = torch.autograd.grad(grad,
                                          self.model.parameters(),
                                          v)
            else:
                raise Exception('Wrong hessian type!')
            
            Hg = self.my_sum(Hg,Hg_)
            
            counter += input.shape[0]
        
        return self.my_div_const(Hg, counter)
    
    
    # computes matrix vector multiplication
    # where the matrix is (H - sum_i val_i vec_i vec_i^T)
    # {val_i}_i and {vec_i}_i are given as input to the class and are usually
    # equal to the top C eigenvalues and eigenvectors
    def mat_vec(self, v):
        Av = self.Hv(v)
        
        for eigvec, eigval in zip(self.vecs, self.vals):
            coeff = eigval * self.my_inner(eigvec, v)
            Av = self.my_sub(Av, self.my_mult_const(eigvec, coeff))
        
        return Av
    
    # compute matrix matrix multiplication by iterating the previous function
    def mat_mat(self, V):
        AV = []
        for v in V:
            AV.append(self.mat_vec(v))
        return AV
    
    # generate a random vector of size #params
    def my_randn(self):
        v_0_l = []
        for param in self.model.parameters():
            Z = torch.randn(param.shape)
            
            if self.double:
                Z = Z.double()
                
            Z = Z.to(self.device)
                
            v_0_l.append(Z)
            
        return v_0_l
    
    # the following functions perform basic operations over lists of parameters
    def my_zero(self):
        return [0 for x in self.my_randn()]
    
    def my_sub(self, X, Y):
        return [x-y for x,y in zip(X,Y)]
    
    def my_sum(self, X, Y):
        return [x+y for x,y in zip(X,Y)]
    
    def my_inner(self, X, Y):
        return sum([torch.dot(x.view(-1), y.view(-1)) for x,y in zip(X,Y)])
    
    def my_mult(self, X, Y):
        return [x*y for x,y in zip(X,Y)]
    
    def my_norm(self, X):
        return torch.sqrt(self.my_inner(X,X))
    
    def my_mult_const(self, X, c):
        return [x*c for x in X]
    
    def my_div_const(self, X, c):
        return [x/c for x in X]
    
    def my_len(self):
        X = self.my_randn()
        return sum([x.view(-1).shape[0] for x in X])
    
    def my_data(self, X):
        return [x.data for x in X]
        
    def my_cpu(self, X):
        return [x.cpu() for x in X]
    
    def my_device(self, X):
        return [x.to(self.device) for x in X]
    
    # compute the minimal and maximal eigenvalue of the linear operator mat_vec
    # this is needed for approximating the spectrum using Lanczos
    def compute_lb_ub(self):
        ritzVal, S, alp, bet = self.Lanczos(self.init_poly_deg)
        
        theta_1 = ritzVal[0]
        theta_k = ritzVal[-1]
        
        s_1 = float(bet[-1]) * float(S[-1,0])
        s_k = float(bet[-1]) * float(S[-1,-1])

        t1 = abs(s_1)
        tk = abs(s_k)
        
        lb = theta_1 - t1
        ub = theta_k + tk
        
        return lb, ub
    
    # approximate the spectrum of the linear operator mat_vec
    def LanczosLoop(self, denormalize=False):
        
        print('Lanczos Method')
        
        lb, ub = self.compute_lb_ub()
        print('Estimated spectrum range:')
        print('[{}\t{}]'.format(lb, ub))
        
        margin = self.spectrum_margin*(ub - lb)
        
        lb -= margin
        ub += margin
        
        print('Spectrum range after adding margin:')
        print('[{}\t{}]'.format(lb, ub))
        
        self.c = (lb + ub)/2
        self.d = (ub - lb)/2
            
        M = self.poly_deg
        
        LB = -1
        UB = 1
        H = (UB - LB) / (M - 1)
        
        kappa = 1.25
        sigma = H / np.sqrt(8 * np.log(kappa))
        sigma2 = 2 * sigma**2
        
        tol = 1e-08
        width = sigma * np.sqrt(-2.0 * np.log(tol))
        
        aa = LB
        bb = UB
        xdos = np.linspace(aa, bb, self.poly_points);
        y = np.zeros(self.poly_points)
        
        ritzVal, S, _, _ = self.Lanczos(self.poly_deg)
        
        ritzVal = (ritzVal - self.c) / self.d
        
        gamma2 = S[0,]**2
                        
        diff = np.expand_dims(ritzVal,-1) - np.expand_dims(xdos,0)
        eigval_idx, pts_idx = np.where(np.abs(diff) < width)
        vals = gamma2[eigval_idx]                                 \
             * np.exp(-((xdos[pts_idx] - ritzVal[eigval_idx])**2) \
             / sigma2)
        np.add.at(y, pts_idx, vals)
        
        scaling = 1.0 / np.sqrt(sigma2 * np.pi)
        y = y*scaling
        
        if denormalize:
            xdos = xdos*self.d + self.c
            y = y/self.d
        
        return xdos, y
    
    # M iteratinos of Lanczos on the linear operator mat_vec
    def Lanczos(self, M):
        v = self.my_randn()
        v = self.my_div_const(v, self.my_norm(v))
        
        alp     = torch.zeros(M)
        bet     = torch.zeros(M)
        
        if self.double:
            alp = alp.double()
            bet = bet.double()
        
        alp = alp.to(self.device)
        bet = bet.to(self.device)
        
        v_prev = None
        
        for j in range(M):
            print('Iteration: [{}/{}]'.format(j+1, M))
                
            sys.stdout.flush()
            
            v_next = self.mat_vec(v)
            
            if j:
                v_next = self.my_sub(v_next, self.my_mult_const(v_prev,bet[j-1]))
                
            alp[j] = self.my_inner(v_next, v)

            v_next = self.my_sub(v_next, self.my_mult_const(v, alp[j]))
            
            bet[j] = self.my_norm(v_next)
            
            v_next = self.my_div_const(v_next, bet[j])
            
            v_prev = v
            v = v_next
            
        B = np.diag(alp.cpu().numpy()) + np.diag(bet.cpu().numpy()[:-1], k=1) + np.diag(bet.cpu().numpy()[:-1], k=-1)
        ritz_val, S = np.linalg.eigh(B)
        
        return ritz_val, S, alp, bet
    
    # compute top-C eigenvalues and eigenvectors using subspace iteration
    def SubspaceIteration(self):
        print('Subspace Iteration')
        
        n = int(self.num_classes)
        
        V = []
        for _ in range(n):
            V.append(self.my_randn())
        
        Q, _ = self.QR(V, n)
        
        for iter in range(self.SSI_iters):
            print('Iteration: [{}/{}]'.format(iter+1, self.SSI_iters))
            sys.stdout.flush()
            
            V = self.mat_mat(Q)
            
            eigvals = [self.my_norm(w) for w in V]
            
            Q, _ = self.QR(V, n)
            
        eigval_density = np.ones(len(eigvals)) * 1/len(eigvals)
        
        return Q, eigvals, eigval_density
    
    # QR decomposition, which is needed for subspace iteration
    def QR(self, A, n):
        Q = []
        R = torch.zeros(n,n)
        
        if self.double:
            R = R.double()
            
        R = R.to(self.device)
        
        for j in range(n):
            v = A[j]
            for i in range(j):
                R[i,j] = self.my_inner(Q[i], A[j])
                v = self.my_sub(v, self.my_mult_const(Q[i], R[i,j]))
            
            R[j,j] = self.my_norm(v)
            Q.append(self.my_div_const(v, R[j,j]))
        
        return Q, R
    
    # compute delta_{c,c'}
    def compute_delta_c_cp(self):
        print("Computing delta_{c,c'}")
        
        if self.hessian_type != 'G':
            raise Exception('Works only for G!')
        
        if self.crit != 'CrossEntropyLoss':
            raise Exception('Works only for cross entropy loss!')

        if self.class_list is not None:
            class_list = self.class_list
        else:
            class_list = [i for i in range(self.num_classes)]
        
        means = []
        counters = []
        for c in class_list:
            means.append([])
            counters.append([])
            for cp in class_list:
                means[-1].append(None)
                counters[-1].append(0)
            
        for idx, batch in enumerate(self.loader, 1):
            print('Iteration: [{}/{}]'.format(idx, len(self.loader)))
            sys.stdout.flush()
            
            input, target = batch[0], batch[1]
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            input = Variable(input)
            target = Variable(target)
            
            f = self.model(input)
            
            prob = F.softmax(f,dim=1)
            
            for idx_c, c in enumerate(class_list):
                
                idxs = (target == c).nonzero()
                
                if len(idxs) == 0:
                    continue
                
                fc = f[idxs.squeeze(-1),]
                probc = prob[idxs.squeeze(-1),]
                
                for idx_cp, cp in enumerate(class_list):
                    # compute delta_{i,c,c'}
                    w = -probc
                    w[:,cp] = w[:,cp] + 1
                    w = w * torch.sqrt(probc[:,[cp]])
                    
                    J = torch.autograd.grad(fc,
                                            self.model.parameters(),
                                            grad_outputs=w,
                                            retain_graph=True)
                    
                    J = self.my_cpu(self.my_data(J))
                    
                    if means[idx_c][idx_cp] is None:
                        means[idx_c][idx_cp] = self.my_zero()
                        
                    means[idx_c][idx_cp] = self.my_sum(means[idx_c][idx_cp], J)
                    counters[idx_c][idx_cp] += fc.shape[0]
        
        for idx_c in range(len(class_list)):
            for idx_cp in range(len(class_list)):
                means[idx_c][idx_cp] = [x/counters[idx_c][idx_cp] for x in means[idx_c][idx_cp]]
            
        return means
    
    # compute G decomposition
    def compute_G_decomp(self, mu_ccp_only=False, mu_only=False, plot_only=False):
        
        # compute delta_{c,c'}
        mu_ccp = self.compute_delta_c_cp()
        
        C = len(mu_ccp)
        
        mu_ccp_flat = []
        for c in range(C):
            for c_ in range(C):
                mu_ccp_flat.append(mu_ccp[c][c_])        
        
        if mu_ccp_only:
            return {'mu_ccp' : mu_ccp}
        
        # compute delta_c
        print("Computing delta_c")
        mu = []
        for c in range(C):
            s = self.my_zero()
            for c_ in range(C):
                if c != c_:
                    s = self.my_sum(s, mu_ccp[c][c_])
            avg = self.my_div_const(s, C-1)
            mu.append(avg)
        
        if mu_only:
            return {'mu' : mu}
        
        # compute distances between {delta_c}_c and {delta_{c,c'}}_{c,c'}
        # (a total of C+C**2 elements)
        # these distances will later be passed to t-SNE
        print("Computing distances for t-SNE plot")
        V = []
        labels = []
        for c in range(C):
            V.append(mu[c])
            labels.append([c])
        for c in range(C):
            for c_ in range(C):
                V.append(mu_ccp[c][c_])
                labels.append([c, c_])
        
        N = C+C**2
        dist = np.zeros([N, N])
        for c in range(N):
            print('Iteration: [{}/{}]'.format(c+1, N))
            for c_ in range(N):
                dist[c,c_] = self.my_norm(self.my_sub(V[c], V[c_]))**2
        
        if plot_only:
            return {'dist'      : dist,
                    'labels'    : labels}
        
        # delta_{c,c}
        mu_cc = []
        for c in range(C):
            mu_cc.append(mu_ccp[c][c])
            
        # compute G0
        print("Computing G0")
        mu_cc_T_mu_cc = np.zeros([C, C])
        for c in range(C):
            for c_ in range(C):
                mu_cc_T_mu_cc[c,c_] = self.my_inner(mu_cc[c], mu_cc[c_]) / C
        G0_eigval, _ = LA.eig(mu_cc_T_mu_cc)
        G0_eigval = sorted(G0_eigval, reverse=True)
        
        # compute G1
        print("Computing G1")
        muTmu = np.zeros([C, C])
        for c in range(C):
            for c_ in range(C):
                muTmu[c,c_] = self.my_inner(mu[c], mu[c_]) * (C-1) / C
        G1_eigval, _ = LA.eig(muTmu)
        G1_eigval = sorted(G1_eigval, reverse=True)

        # compute G1+2
        print("Computing G1+2")
        mu_ccp_T_mu_ccp = np.zeros([C**2, C**2])
        for c in range(C**2):
            for c_ in range(C**2):
                mu_ccp_T_mu_ccp[c,c_] = self.my_inner(mu_ccp_flat[c], mu_ccp_flat[c_]) / C
        G12_eigval, _ = LA.eig(mu_ccp_T_mu_ccp)
        G12_eigval = sorted(G12_eigval, reverse=True)
                
        # compute G_2
        print("Computing G2")
        nu = []
        for c in range(C):
            nu.append([])
            for c_ in range(C):
                nu[c].append(self.my_sub(mu_ccp[c][c_], mu[c]))
            
        nu_flat = []
        for c in range(C):
            for c_ in range(C):
                if c != c_:
                    nu_flat.append(nu[c][c_])
        
        gram_nu_flat = np.zeros([C*(C-1), C*(C-1)])
        for c in range(C*(C-1)):
            for c_ in range(C*(C-1)):
                gram_nu_flat[c,c_] = self.my_inner(nu_flat[c], nu_flat[c_]) / C
        G2_eigval, _ = LA.eig(gram_nu_flat)
        G2_eigval = sorted(G2_eigval, reverse=True)
        
        # density is 1/(number of eigenvalues)
        G0_eigval_density  = np.ones(len(G0_eigval)) * 1/len(G0_eigval)
        G1_eigval_density  = np.ones(len(G1_eigval)) * 1/len(G1_eigval)
        G12_eigval_density = np.ones(len(G12_eigval)) * 1/len(G12_eigval)
        G2_eigval_density  = np.ones(len(G2_eigval)) * 1/len(G2_eigval)
        
        res = {'mu_ccp'             : mu_ccp,
               'mu_ccp_flat'        : mu_ccp_flat,
               'mu'                 : mu,
               'nu'                 : nu,
               'nu_flat'            : nu_flat,
               'G0_eigval'          : G0_eigval,
               'G0_eigval_density'  : G0_eigval_density,
               'G1_eigval'          : G1_eigval,
               'G1_eigval_density'  : G1_eigval_density,
               'G2_eigval'          : G2_eigval,
               'G2_eigval_density'  : G2_eigval_density,
               'G12_eigval'         : G12_eigval,
               'G12_eigval_density' : G12_eigval_density,
               'dist'               : dist,
               'labels'             : labels,
                }
        
        return res
    
    


    