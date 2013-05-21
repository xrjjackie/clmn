//
//  utilities.h
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 3/20/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#ifndef GaussianMarkovNetworks_utilities_h
#define GaussianMarkovNetworks_utilities_h

#include <iostream>
#include <math.h>
#include <vector>

struct message {
    double val;
    double iv;
    double h;
};

double logsumexp(double nums[], size_t ct);
double logsumexp(const std::vector<double>& nums);
double weighted_average(double logweights[], double vals[], size_t n);

#endif
