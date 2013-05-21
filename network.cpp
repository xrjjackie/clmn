//
//  network.cpp
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 2/29/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#include "network.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "glog/logging.h"
#include "gflags/gflags.h"

using namespace std;

DEFINE_double(laplacian_epsilon, 1.0, "epsilon added to the diagonal of graph laplacian");

int compare (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

void Network::load_data_from_file(string filepath) {
    ifstream fin((filepath + "_node_attributes.txt").c_str());
    ifstream flinks((filepath + "_links.txt").c_str());
    int b;
    fin >> num_instances_ >> feature_dim_ >> b;
    CHECK(b==2) << "Must be a binary classification problem! #classes read: " << b;
    adjacency_.reset(new SparseMatrix<double, RowMajor> (num_instances_, num_instances_));
    attributes_.reset(new boost::scoped_array<double>[num_instances_]);
    labels_.reset(new int[num_instances_]);
    degrees_.reset(new int[num_instances_]);
    boost::scoped_array<int> linked; 
    for (int i = 0; i < num_instances_; ++i) {
        adjacency_->startVec(i);
        double y;
        fin >> y;        
        if (y > 0) labels_[i] = 1;
        else labels_[i] = -1;
        attributes_[i].reset(new double[feature_dim_]);
        for (int j = 0; j < feature_dim_; ++j) {
            fin >> attributes_[i][j];
        }
        flinks >> degrees_[i];
        linked.reset(new int[degrees_[i]]);
        for (int j = 0; j < degrees_[i]; ++j) {
            int k;
            flinks >> k;
            linked[j] = k;
        }
        qsort(linked.get(), degrees_[i], sizeof(int), compare);
        for (int j = 0; j < degrees_[i]; ++j) {
            int k = linked[j];
            if (k!=i) adjacency_->insertBack(i,k) = 1;
        }        
    }
    fin.close();
    flinks.close();
    adjacency_->finalize();
}

int Network::get_instance_label(int index) const {
    return labels_[index];
}

double* Network::get_instance_attributes(int index) const {
    return attributes_[index].get();
}

SparseMatrix<double, RowMajor> Network::get_graph_laplacian() const {
    SparseMatrix<double, RowMajor> diag(num_instances_, num_instances_);
    for (int i = 0; i < num_instances_; ++i) {
        diag.startVec(i);
        diag.insertBack(i,i) = degrees_[i] + FLAGS_laplacian_epsilon;
    }
    diag.finalize();
    SparseMatrix<double, RowMajor> laplacian(diag - (*adjacency_));
    return laplacian;
}
