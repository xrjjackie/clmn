//
//  lgmn.h
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 3/20/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#ifndef GaussianMarkovNetworks_lgmn_h
#define GaussianMarkovNetworks_lgmn_h

#include "boost/scoped_array.hpp"
#include "boost/scoped_ptr.hpp"
#include "utilities.h"
#include "network.h"
#include "logistic.h"
#include "randomc.h"

class LatentGaussMarkovNet {
    boost::scoped_array<int> observed_labels_;
    Network* data_;
    boost::scoped_ptr<Logistic> logistic_;
    int num_nodes_;
    boost::scoped_array<double> laplacian_diag_;    
    boost::scoped_array<double> m_by_ep_;
    boost::scoped_array<double> v_by_ep_;
    pair<double, double> update_by_likelihood(int node, double s_iv, double s_h, double m_all, double v_all);
    boost::scoped_ptr<CRandomMersenne> random_;
    boost::scoped_array<map<int, message> > messages_;
public:
    LatentGaussMarkovNet(Network* data);
    void set_labeled(const vector<int>& labeled);
    void expectation_propagation();
    void predict(vector<double>& prediction);
};


#endif
