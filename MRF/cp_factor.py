
# gpu = False
gpu = True

from .domain import Domain
import scipy
from collections import Iterable
import time

import numpy as np
import cupy as cp

if gpu:
    xp = cp
else:
    xp = np

# debug
debug_time_cost = 0

class Factor:
    def __init__(self, domain: Domain, values, xp=xp):
        # assert(xp==cp)
        # print(xp)
        self.domain = domain
        if cp.get_array_module(values) != xp:
            values = xp.array(values)
        self.values = values
        self.size = len(self.domain)

        if not tuple(domain.shape) == self.values.shape:
            print(domain)
            print(tuple(domain.shape), self.values.shape)
            assert(domain.shape == self.values.shape)
    
    def to_cpu(self):
        if cp.get_array_module(self.values) != np:
            self.values = cp.asnumpy(self.values)
        return self
    
    def mov_to_cpu(self):
        if cp.get_array_module(self.values) != np:
            values = cp.asnumpy(self.values)
        else:
            values = self.values.copy()
        return Factor(self.domain.copy(), values, np)

    def to_gpu(self):
        if cp.get_array_module(self.values) != cp:
            self.values = cp.array(self.values)
        return self

    def __add__(self, parameter):
        if not isinstance(parameter, Factor):
            return Factor(self.domain, self.values + parameter, cp.get_array_module(self.values))
        domain = self.domain + parameter.domain
        add1 = self.expand(domain)
        add2 = parameter.expand(domain)
        return Factor(domain, add1.values + add2.values, cp.get_array_module(self.values))

    def __mul__(self, parameter):
        if not isinstance(parameter, Factor):
            return Factor(self.domain, self.values * parameter, cp.get_array_module(self.values))
        domain = self.domain + parameter.domain
        mul1 = self.expand(domain)
        mul2 = parameter.expand(domain)
        return Factor(domain, mul1.values * mul2.values, cp.get_array_module(self.values))

    def __rmul__(self, parameter):
        return self.__mul__(parameter)

    def __imul__(self, parameter):
        self.values *= parameter
        return self

    def __truediv__(self, parameter):
        if isinstance(parameter, Factor):
            parameter = Factor(parameter.domain, 1/parameter.values, cp.get_array_module(self.values))
            return self * parameter
        return self * (1/parameter)
    
    def __sub__(self, parameter):
        if isinstance(parameter, Factor):
            domain = self.domain + parameter.domain
            sub1 = self.expand(domain)
            sub2 = parameter.expand(domain)
            
            return Factor(domain, sub1.values - sub2.values, cp.get_array_module(self.values))
        else:
            return Factor(self.domain, self.values - parameter, cp.get_array_module(self.values))

    def sum(self):
        return cp.get_array_module(self.values).sum(self.values)

    @staticmethod
    def zeros(domain, xp=xp):
        return Factor(domain, xp.zeros(domain.shape), xp)
    
    @staticmethod
    def full(domain, fill_value, xp=xp):
        return Factor(domain, xp.full(fill_value=fill_value, shape=domain.shape), xp)

    def expand(self, domain):
        if len(domain) == len(self.domain):
            return self
        assert(set(self.domain.dict.keys()) <= set(domain.dict.keys()))
        shape = self.domain.shape + [1] * (len(domain) - len(self.domain))
        
        index_list = domain.index_list(self.domain)

        # print(domain.attr_list, domain.shape)
        # print(shape)
        # print(self.domain.attr_list, self.domain.shape, self.values.shape)

        values = self.values.reshape(shape)
        values = cp.get_array_module(self.values).moveaxis(values, range(len(self.domain)), index_list)
        values = cp.get_array_module(self.values).broadcast_to(values, domain.shape)

        return Factor(domain, values, cp.get_array_module(self.values))
    
    # rescale the factor along a given variable set
    def scale_along_var(self, var_set, total_array):
        start_time = time.time()

        xp = cp.get_array_module(self.values)
        # assert(xp == cp.get_array_module(total_array))
        # assert(set(var_set) <= set(self.domain.attr_list))
        # self.values = xp.array(np.random.random(size=self.values.shape))
        # print(xp)
        # print(self.domain)
        # print(self.domain.project(var_set).shape)
        # print(total_array.shape)
        
        hist_total = self.project(self.domain.project(var_set)).values
        scale_ratio = xp.divide(total_array, hist_total)

        axis_list = self.domain.index_list(sorted(list(var_set)))
        shape = [1] * len(self.domain)
        for axis in axis_list:
            shape[axis] = self.domain.shape[axis]
        scale_ratio = scale_ratio.reshape(shape)

        values = xp.multiply(self.values, scale_ratio)

        # print(total_array.shape)
        # print(self.values.shape)
        # print(scale_ratio.shape)
        # print(values.shape)

        # print(values[2, 2, 1, 4, 1, 2], self.values[2, 2, 1, 4, 1, 2]*scale_ratio[0, 0, 0, 0, 1, 2],\
        #     self.values[2, 2, 1, 4, 1, 2], scale_ratio[0, 0, 0, 0, 1, 2])
        # print(values[4, 1, 5, 9, 1, 2], self.values[4, 1, 5, 9, 1, 2]*scale_ratio[0, 0, 0, 0, 1, 2],\
        #     self.values[4, 1, 5, 9, 1, 2], scale_ratio[0, 0, 0, 0, 1, 2])
        # print(values[3, 7, 2, 7, 1, 2], self.values[3, 7, 2, 7, 1, 2]*scale_ratio[0, 0, 0, 0, 1, 2],\
        #     self.values[3, 7, 2, 7, 1, 2], scale_ratio[0, 0, 0, 0, 1, 2])

        # print(xp.sum(values, axis=[0, 1, 2, 3]))
        # print(total_array)
        # exit(0)

        global debug_time_cost
        debug_time_cost += time.time() - start_time
        # print(debug_time_cost)
        return Factor(self.domain.copy(), values, xp)

    def conditional_value(self, p_cond_attr, p_cond_value):
        # print(self.domain.attr_list, p_cond_attr, p_cond_value)
        cond_attr = []
        cond_value = []
        for i in range(len(p_cond_attr)):
            if p_cond_attr[i] in self.domain.attr_list:
                cond_attr.append(p_cond_attr[i])
                cond_value.append(p_cond_value[i])

        # print(cond_attr, cond_value)

        index_list = self.domain.index_list(cond_attr)
        temp_value = self.values.copy()
        shape = list(temp_value.shape)
        # print(f'original shape: {shape}')

        # print(index_list)
        for i in range(len(index_list)):
            axis = index_list[i]

            shape[axis] = 1

            temp_value = cp.take(temp_value, cond_value[i], axis=axis)
            # print(temp_value.shape, shape)
            temp_value = temp_value.reshape(shape)

        shape = [item for item in shape if item != 1]
        temp_value = temp_value.reshape(shape)

        return Factor(self.domain.invert(cond_attr), temp_value)

    def logsumexp(self, attr_set=None):
        if attr_set == None or len(attr_set) == 0:
            xp = cp.get_array_module(self.values)
            if xp == cp:
                values = xp.exp(self.values)
                values = xp.sum(values)
                values = xp.log(values)
                return values
            else:
                return scipy.special.logsumexp(self.values)
        assert(set(attr_set) <= set(self.domain.attr_list))
        sum_attr = list(set(self.domain.attr_list) - set(attr_set))
        sum_attr = tuple(self.domain.index_list(sum_attr))
        if cp.get_array_module(self.values) == cp:
            values = cp.exp(self.values)
            values = cp.sum(values, axis=sum_attr)
            values = cp.log(values)
        else:
            values = scipy.special.logsumexp(self.values, axis=sum_attr)
        return Factor(self.domain.project(attr_set), values, cp.get_array_module(self.values))

    def exp(self):
        return Factor(self.domain, cp.get_array_module(self.values).exp(self.values), cp.get_array_module(self.values))

    # project to an attr set while keeping the original attr orders
    def project(self, domain):
        if not isinstance(domain, Domain):
            if not isinstance(domain, Iterable):
                domain = [domain]
            domain = self.domain.project(domain)
        # print('projecting:')
        # print(self.domain)
        # print(domain)
        assert(set(domain.attr_list) <= set(self.domain.attr_list))
        new_domain = self.domain.invert(domain)
        index_list = tuple(self.domain.index_list(new_domain))

        values = cp.get_array_module(self.values).sum(self.values, axis=index_list)
        return Factor(domain, values, cp.get_array_module(self.values))

    def copy(self):
        return Factor(self.domain, self.values.copy(), cp.get_array_module(self.values))

    def moveaxis(self, attr_list):
        new_domain = self.domain.moveaxis(attr_list)
        index_list = tuple(new_domain.index_list(self.domain))
        values = cp.get_array_module(self.values).moveaxis(self.values, range(len(self.domain)), index_list)
        return Factor(new_domain, values, cp.get_array_module(self.values))

    def log(self):
        return Factor(self.domain, cp.log(self.values + 1e-100))

class Potential(dict):
    def __init__(self, factor_dict):
        dict.__init__(self, factor_dict)
        
    def __sub__(self, potential):
        assert(len(self) == len(potential))
        ans = {clique: self[clique] - potential[clique] for clique in self}
        return Potential(ans)

    def __add__(self, potential):
        assert(len(self) == len(potential))
        ans = {clique: self[clique] + potential[clique] for clique in self}
        return Potential(ans)

    def __mul__(self, parameter):
        return Potential({clique: parameter*self[clique] for clique in self})

    def __rmul__(self, parameter):
        return self.__mul__(parameter)

    def __imul__(self, parameter):
        for clique in self:
            self[clique] *= parameter
        return self

    def dot(self, potential):
        for clique in self:
            xp = cp.get_array_module(self[clique].values)
            break
        return sum(xp.sum((self[clique] * potential[clique]).values) for clique in self)

    def copy(self):
        return Potential({clique: self[clique].copy() for clique in self})
    
    def mov_to_cpu(self):
        new_potential = Potential({})
        for item in self:
            new_potential[item] = self[item].mov_to_cpu()
        return new_potential

    def to_cpu(self):
        for item in self:
            self[item].to_cpu()

    def to_gpu(self):
        for item in self:
            self[item].to_gpu()

    @staticmethod
    def l2_marginal_loss(marginal_potential1, marginal_potential2):
        gradient = marginal_potential1 - marginal_potential2
        loss = 1/2 * gradient.dot(gradient)
        return loss.item(), gradient

    def fill(self, value):
        for clique in self:
            self[clique].values.fill(value)
        return self

    def clip(self, abs_value):
        for clique in self:
            idx = self[clique].values > abs_value
            self[clique].values[idx] = abs_value

            idx = self[clique].values < -abs_value
            self[clique].values[idx] = -abs_value