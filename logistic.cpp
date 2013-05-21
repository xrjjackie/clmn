//
//  logistic.cpp
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 2/29/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#include "logistic.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "nlopt.hpp"

DEFINE_double(logistic_scale, 1.0, "scale parameter in the logistic model");
DEFINE_double(logistic_regularize, 1.0, "regularization weight for the logistic model");

Logistic::Logistic(int feature_dim) {
    dim_ = feature_dim + 1;
    scale_ = FLAGS_logistic_scale;
    for (int i = 0; i < dim_; ++i) parameters_.push_back(0.0);
}

void Logistic::clear_data() {
    x_.clear();
    y_.clear();
}

void Logistic::add_instance(double x[], int y) {
    vector<double> vx(dim_);
    for (int i = 0; i < dim_-1; ++i) vx[i] = x[i];
    vx[dim_ - 1] = 1.0;
    x_.push_back(vx);
    y_.push_back(y);
}

double dotproduct(vector<double> x, vector<double> y) {
    double s = 0;
    CHECK(x.size() == y.size());
    for (int i = 0; i < x.size(); ++i) {
        s += x[i] * y[i];
    }
    return s;    
}

double Logistic::get_mean(int index) {
    return dotproduct(parameters_, x_[index]);
}

double negloglik(const std::vector<double> &w, std::vector<double> &gradient, void* data) {
    vector<vector<double> >* xy = (vector<vector<double> >*) data;
//    LOG(INFO) << "w: " << w[0] << " " << w[1] << " " << w[2];
    // Prior likelihood                                                                                                                                                                            
    double l = (FLAGS_logistic_regularize / 2.0) * dotproduct(w, w);
    for (int j = 0; j < gradient.size(); ++j) gradient[j] = FLAGS_logistic_regularize * w[j];
    for (int i = 0; i < xy->size(); ++i) {
        double v = dotproduct(w, (*xy)[i]) / FLAGS_logistic_scale;
        if (v < -30) l += -v;
        else l += log(1.0 + exp(-v));              
        if (v < 30) {
            for (int j = 0; j < gradient.size(); ++j) {
                gradient[j] -= (*xy)[i][j]/(exp(v) + 1.0)/FLAGS_logistic_scale;
            }
        }
    }   
//    LOG(INFO) << " nll: " << l << "gradient: " << gradient[0] << " " << gradient[1] << " " << gradient[2];
    return l;
}

void Logistic::train() {
    nlopt::opt optobj(nlopt::LD_LBFGS, dim_);
    vector<vector<double> > data(x_.size());
    for (int i = 0; i < x_.size(); ++i) {
        vector<double> xy(x_[i]);
        double yi = y_[i];
        for (int j = 0; j < x_[i].size(); ++j) xy[j] *= yi;
        data[i] = xy;
    }
    optobj.set_min_objective(negloglik, &data);
    optobj.set_xtol_rel(1e-5);
    optobj.set_ftol_rel(1e-3);
    double value;
    try {
        nlopt::result result = optobj.optimize(parameters_, value);
        LOG_IF(ERROR, result<0) << "optimization unsuccessful! Error code: " << result;
    } catch (exception& e) {
        LOG(ERROR) << "NLOPT exception: " << e.what();
    }
}

int Logistic::get_num_instances() {
    return x_.size();
}
