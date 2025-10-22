# pce_core.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import legendre, hermitenorm
from scipy import stats
import warnings
import math
from timeit import default_timer as timer
from joblib import Parallel, delayed
import pce_utils

# Set plotting style
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

class PolyChaos():

    def __str__(self):
        kmax = 5
        msg = ""
        
        msg += "Polynomial Chaos Expansion\n"
        msg += "--------------------------\n"
        msg += f"    dimensions: {self.dim}\n"
        msg += f"    order: {self.order}\n"
        
        msg += "    distrib: ["
        for k, ele in enumerate(self.distrib):
            if k > kmax - 1:
                msg += ' ... '
                break
            elif k == len(self.distrib)  - 1:
                msg += ele.upper()
            else:
                msg += ele.upper() + " , "
        msg += "]\n"

        msg += "    param: ["
        for k, ele in enumerate(self.param):
            if k > kmax - 1:
                msg += ' ... '
                break
            elif k == len(self.param)  - 1:
                msg += str(ele)
            else:
                msg += str(ele) + " , "
        msg += "]\n"

        msg += "    coeff: ["
        if self.coeff.size == 0:
            msg += " to be computed "
        else:
            for k, ele in enumerate(self.coeff):
                if k > kmax - 1:
                    msg += ' ... '
                    break
                elif k == len(self.param) - 1:
                    msg += str(ele)
                else:
                    msg += str(ele) + " , "
        msg += "]\n"
        
        return msg

    def __init__(self, order, distrib, param):

        if len(distrib) != len(param):
            raise ValueError('distrib and param not the same length')
        
        self.dim = len(distrib)
        self.order = order
        self.distrib = distrib
        self.param = param
        self.coeff = np.empty(0)
        self.grid = None
        self.mu = None
        self.sigma = None

        (self.nt,
         self.multi_index,
         self.basis) = self.create_instance(order, distrib)
    

    def create_instance(self, order, distrib):

        # generate multi index
        multi_index, nt = self.generate_multi_index(order)

        # create multivariate polynomials basis 
        def basis(index, eps):

            if index > (nt - 1):
                raise ValueError(f'max index possible is nt-1={self.nt-1}')
            if isinstance(eps, (list, tuple)):
                eps = np.array(eps)

            i = self.multi_index[index]
            
            y = np.ones(eps.shape[0])
           
            for k, dist in enumerate(distrib):
                if dist.upper() == 'U':
                    y = y * np.polyval(legendre(i[k]), eps[..., k])
                elif dist.upper() == 'N':
                    y = y * np.polyval(hermitenorm(i[k]), eps[..., k])
            
            return y
        
        return nt, multi_index, basis

        
    def generate_multi_index(self, order):
        """Generate the total-order multi-index set (sum of indices <= order)."""
        index = np.empty((0, self.dim), dtype=int)
        cnt = 0
        for val in range(order + 1):
            k = np.zeros(self.dim,dtype=int)
            khat = np.ones_like(k) * val    
            p = 0
            while k[self.dim - 1] <= val:
                if k[p] > khat[p]:
                    if (p + 1) != self.dim:
                        k[p] = 0
                        p += 1
                else:
                    khat[:p] = khat[p] - k[p]
                    k[0] = khat[0]
                    p = 0
                    cnt += 1
                    index = np.append(index, k.reshape(1,-1), axis=0)
                k[p] = k[p] + 1
        return index, cnt


    def norm_factor(self, multi_index):
        """Compute the inverse of the squared L2-norm for the orthogonal polynomial $1/\mathbb{E}[\Psi_i^2]$."""
        factor = 1
        for k, index in enumerate(multi_index):
            if self.distrib[k].upper() == 'U':
                factor = factor * (2 * index + 1) / 1
            elif self.distrib[k].upper() == 'N':
                factor = factor / math.factorial(index)
        
        return factor

    
    def spectral_projection(self, fun, level,):
        ''' 
        create sparse grid. Not called when employing least squares PCE
        '''
        # ------------------
        t1 = timer()
        print("* generation of smolyak sparse grid ... ", end=' ', flush=True)
        #
        self.grid = pce_utils.PceSmolyakGrid(self, level)

        # evaluate function at unique points
        # ----------------------------------
        
        unique_y = fun(self.grid.unique_x)
        #
                
        # evaluate function at all points
        # -------------------------------
        print(f"* build complete function output ({self.grid.x.shape[0]} points) ... ", end=' ', flush=True)
            #
        y = unique_y[self.grid.inverse_index, ...]
        
        # coefficient computation
        # -----------------------
        print("* coefficient computation ... ", end=' ', flush=True)
        
        # function to compute the k-th coefficient
        def compute_k_coeff(k):
            factor = self.norm_factor(self.multi_index[k])
            return factor * np.sum(y.flatten() * self.basis(k, self.grid.eps).flatten() * self.grid.weight.flatten())
        # parallel computation of all coefficients
        coeff = Parallel(n_jobs=-1, verbose=0)(map(delayed(compute_k_coeff), range(self.nt)))
        self.coeff = np.array(coeff)
    

    def create_latin_hypercube_samples(self,sp_num,rd_state=None):
        """Generate input samples using Latin Hypercube Sampling (LHS)."""
        dim = self.dim
        sample_points = np.zeros((sp_num, dim))
        rng = np.random.RandomState(rd_state) if rd_state is not None else np.random

        # create the latin hyper-cube 
        randoms = rng.random(sp_num * dim).reshape((dim, sp_num))
        for dim_ in range(dim):
            perm = rng.permutation(sp_num)  # pylint: disable=no-member
            randoms[dim_] = (perm + randoms[dim_]) / sp_num

        for k, (dist,param) in enumerate(zip(self.distrib,self.param)):
        # rescale          
        # from [0, 1] to [a,b] for uniform distribution
        # from [0, 1] to [nu,sigma] for normal distribution 
            if dist.upper() == 'N':
                nu,sigma = param
                std_nm = stats.norm.ppf(randoms.T[:,k])
                sample_points[:,k] = nu + sigma*std_nm
            elif dist.upper() == 'U':
                min_v,max_v = param
                sample_points[:,k] = min_v + (max_v-min_v)*randoms.T[:,k]
        
        print(f'Latin Hypercube sampling: {sp_num} samples in {dim} dimensions created.\n')
        return sample_points
    

    def regression_fit(self,  points, y_samples):
        """Fit PCE coefficients using Least-Squares Regression (LSR).""" 
        Ns = points.shape[0]
        if Ns < self.nt:
            raise ValueError(f'number of points ({Ns}) must be >= number of basis functions ({self.nt})\nIncrease Ns or decrease PCE order.')
        if y_samples.shape[0] != Ns:
            raise ValueError(f'number of points {points.shape} and samples {y_samples.shape} must be the same')

        # change of limited range
        std_points = np.zeros_like(points)
        for k, (dist,param) in enumerate(zip(self.distrib,self.param)):
            if dist.upper() == 'U':
                a, b = param
                std_points[:,k] = (2 * points[:,k]-a-b)/(b-a)
            elif dist.upper()=='N':
                nu,sigma = param
                std_points[:,k] = (points[:,k]-nu)/sigma

        # build the regression matrix
        A = np.zeros((Ns,self.nt))
        for k in range(self.nt):
            A[:,k] = self.basis(k,std_points)

        # compute the coefficients
        coeff, residuals, rank, singular_vals = np.linalg.lstsq(A, y_samples, rcond = None)

        self.coeff = coeff.flatten()

        if rank<min(A.shape):
            warnings.warn(f'Regression matrix is rank deficient.Rank={rank}, expected min={min(A.shape)}')

        print(f'Regression-Fit: PCE order={self.nt},residuals(L2)={np.sqrt(residuals/Ns)}\n')

    def compare_on_realData(self,input_lhs,model_function,real_samples=None,verbose = 'y'):
        """Validate PCE model on LHS samples and optionally on a 'real' dataset."""
        y_samples_lhs = model_function(input_lhs).squeeze()
        output_pce_lhs = self.evaluate(input_lhs)

        if verbose == 'y':
            figs, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # LHS Sample Comparison
            axes[0].plot(y_samples_lhs, output_pce_lhs, 'o', color='#c20078', markerfacecolor='none')
            axes[0].plot(axes[0].get_xlim(), axes[0].get_xlim(), 'k--')
            axes[0].set_xlabel('Model Output (LHS)')
            axes[0].set_ylabel('PCE Output (LHS)')
            axes[0].set_title('PCE on Training Samples')
            
            pce_output_real, sr_output_real = None, None
            
            # Real Sample Comparison
            if real_samples is not None:
                pce_output_real = self.evaluate(real_samples)
                sr_output_real = model_function(real_samples).squeeze()
                
                axes[1].plot(sr_output_real, pce_output_real, 'o', color='#00c251', markerfacecolor='none', )
                axes[1].plot(axes[1].get_xlim(), axes[1].get_xlim(), 'k--')
                axes[1].set_xlabel('Model Output')
                axes[1].set_ylabel('PCE Output')
                axes[1].set_title('PCE on Real Data Points')

                total_samples = len(sr_output_real)
                print(f"Validation on Real Data: {total_samples} points.")
            
            plt.tight_layout()
            plt.show()

        return [output_pce_lhs, y_samples_lhs, pce_output_real, sr_output_real]



    def norm_fit(self,):
        """Compute statistical moments (mean \mu and variance \sigma^2) from PCE coefficients."""        
        # evaluation of the mean
        self.mu = self.coeff[0]
        
        # evaluation of the standard deviation
        c_quad = self.coeff[1:] ** 2
        psi_quad = np.array([1/self.norm_factor(k) for k in self.multi_index[1:]])
        self.sigma = np.sqrt(np.sum(c_quad * psi_quad))

        print(f"Mean : {self.mu:.4f}, Std. Dev. : {self.sigma:.4f}")

    
    def sobol(self, index):
        """
        SOBOL computes the Sobol' indices for a given PCE expansion.

        Paramaters
        ----------
        index (list) :
            list with index, e.g. [1,2] for having S12, [[1], [1,2,3]] for
            having S1 and S123
        
        """
        
        # check precedences
        if self.sigma is None:
            raise ValueError("norm_fit() must be called before sobol() can be executed")

        # handle input type
        if not isinstance(index[0], (list, tuple)):
            index = [index]
        
        sobol = []
        for idx in index:
            # create complementary index
            zero_based_index = [k - 1 for k in idx]
            all_index = np.array(range(len(self.distrib)))
            other_index = np.setdiff1d(all_index, zero_based_index)

            # find elements
            coeff_index = np.array([True] * (self.multi_index.shape[0] - 1))
            for ele in zero_based_index:
                coeff_index = coeff_index * (self.multi_index[1:, ele] != 0)
            for ele in other_index:
                coeff_index = coeff_index * (self.multi_index[1:, ele] == 0)
            
            # computation of the index
            c_quad = self.coeff[1:] ** 2
            psi_quad = np.array([1/self.norm_factor(k) for k in self.multi_index[1:]])
            sobol.append(np.sum(c_quad[coeff_index] * psi_quad[coeff_index]) / (self.sigma ** 2))
        
        # prepare output type
        if len(sobol) == 1:
            sobol = sobol[0]
        
        return sobol


    def evaluate(self, points):
        """Evaluate the PCE model at given physical input points."""        
        if self.coeff.shape == (0, ):
            import warnings
            warnings.warn('PCE coeffiecients are not yet computed.')
        else:            
            # initialization
            std_points = np.zeros_like(points)

            # change of coordinates
            for k, (dist, param) in enumerate(zip(self.distrib, self.param)):
                if dist.upper() == 'U':
                    std_points[:, k] = (param[0] + param[1] - 2 * points[:, k]) / (param[0] - param[1])
                elif dist.upper() == 'N':
                    std_points[:, k] = (points[:,k] -  param[0]) / param[1]
            
            for k in range(self.nt):
                if k == 0:
                    y = self.coeff[k] * self.basis(k, std_points)
                else:
                    y += self.coeff[k] * self.basis(k, std_points)
            
            return y.squeeze()

