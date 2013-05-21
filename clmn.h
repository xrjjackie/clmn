//
//  clmn.h
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 3/1/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#ifndef GaussianMarkovNetworks_clmn_h
#define GaussianMarkovNetworks_clmn_h

#include <utility>
#include <map>
#include "boost/scoped_array.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/math/distributions/normal.hpp"
#include "network.h"
#include "logistic.h"
#include "randomc.h"
#include "utilities.h"

class CopulaLatentMarkovNet {
    boost::scoped_array<double> gmn_marginal_scale_;
    boost::scoped_array<int> observed_labels_;
    boost::scoped_array<double> cut_;
    Network* data_;
    boost::scoped_ptr<Logistic> logistic_;
    int num_nodes_;
    boost::scoped_array<double> laplacian_diag_;    
    boost::scoped_array<double> m_by_ep_;
    boost::scoped_array<double> v_by_ep_;
    double compute_loglikelihoodpart(double ts, const boost::math::normal& nd, double msigma);
    pair<double, double> metropolis_hasting(int node, double s_iv, double s_h);
    boost::scoped_ptr<CRandomMersenne> random_;
    boost::scoped_array<map<int, message> > messages_;
public:
    CopulaLatentMarkovNet(Network* data);
    void set_labeled(const vector<int>& labeled);
    void expectation_propagation();
    void predict(vector<double>& prediction);
    void lr_predict(vector<double>& prediction);
};

#endif
