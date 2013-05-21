//
//  logistic.h
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 2/29/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#ifndef GaussianMarkovNetworks_logistic_h
#define GaussianMarkovNetworks_logistic_h

#include <vector>

using namespace std;

class Logistic {
    int dim_;
    vector<double> parameters_;
    vector< vector<double> > x_;
    vector<double> y_;
public:
    double scale_;
    Logistic(int feature_dim);
    void clear_data();
    void add_instance(double x[], int y);
    double get_mean(int index);
    void train();
    int get_num_instances();
};

#endif
