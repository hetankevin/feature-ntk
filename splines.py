
# This is a demonstration of a tensor spline or multivariate spline based regressor using scikit-learn
# 
# This script is adapted from https://gist.github.com/MMesch/35d7833a3daa4a9e8ca9c6953cbe21d4
# which provides a spline based transformer performed independently for each feature. This
# is adapted for additively separable functions (i.e., f(x,y) = g(x) + h(y)).
# 
# We propose here to construct a basis of ND tensor splines or multivariate splines, thus
# able to fit correlation between features.
# 
# We demonstrate the method on a small example.

# MIT License

# Copyright (c) 2021 Eric Chassande-Mottin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from itertools import product

import pandas as pd
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from scipy.interpolate import splrep, splev

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Tensor spline basis and features -- Transformer

class TensorSpline(object):
    def __init__(self, knots, degree=3, periodic=False):
        self.knots = knots
        self.degree = degree
        self.periodic = periodic
        
    def bases(self):
        """Returns a list of tuple (t,c,k) containing the vector of knots, the B-spline 
            coefficients, and the degree of the spline for each feature axis."""

        bases = {}
        for feature, knots in self.knots.items():
            knots_proc, coeffs_proc, degree = splrep(knots, 
                                                     numpy.zeros(len(knots)), 
                                                     k=self.degree, 
                                                     per=self.periodic)
            basis = []
            for ispline in range(len(self.knots[feature])):
                coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(len(coeffs_proc))]
                basis.append((knots_proc, coeffs, degree))                
                bases[feature] = basis
                
        return bases

class TensorSplineFeatures(TransformerMixin):
    def __init__(self, knots, degree=3, periodic=False):
        self.splines = TensorSpline(knots, degree=3, periodic=False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ Computes the value of the spline tensor basis for all data points
        """
        bases = self.splines.bases()
        features = []
        for t in product(*bases.values()):
            features.append(math.prod([splev(X[k], s) for k, s in zip(bases.keys(), t)]))
        return numpy.array(features).T
