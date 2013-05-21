//
//  network.h
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 2/29/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#ifndef GaussianMarkovNetworks_network_h
#define GaussianMarkovNetworks_network_h

#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif

#include <string>
#include <vector>
#include "Eigen/Sparse"
#include "boost/scoped_ptr.hpp"
#include "boost/scoped_array.hpp"

using namespace std;
using namespace Eigen;

class Network {
    boost::scoped_ptr<SparseMatrix<double, RowMajor> > adjacency_;
    boost::scoped_array< boost::scoped_array<double> > attributes_;
    boost::scoped_array<int> labels_;
    boost::scoped_array<int> degrees_;
public:
    void load_data_from_file(string filepath);
    SparseMatrix<double, RowMajor> get_graph_laplacian() const;
    int get_instance_label(int index) const;
    double* get_instance_attributes(int index) const;
    int feature_dim_;
    int num_instances_;
};

#endif
