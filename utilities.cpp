//
//  utilities.cpp
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 3/20/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#include <iostream>
#include "utilities.h"

double logsumexp(double nums[], size_t ct) {
    double max_exp = nums[0], sum = 0.0;
    size_t i;
    
    for (i = 1 ; i < ct ; i++)
        if (nums[i] > max_exp)
            max_exp = nums[i];
    
    for (i = 0; i < ct ; i++)
        sum += exp(nums[i] - max_exp);
    
    return log(sum) + max_exp;
}

double logsumexp(const std::vector<double>& nums) {
    double max_exp = nums[0], sum = 0.0;
    
    for (std::vector<double>::const_iterator it = nums.begin(); it != nums.end(); ++it)
        if ((*it) > max_exp) max_exp = (*it);
    
    for (std::vector<double>::const_iterator it = nums.begin(); it != nums.end(); ++it)
        sum += exp((*it) - max_exp);
    
    return log(sum) + max_exp;
}

double weighted_average(double logweights[], double vals[], size_t n) {
    double logsw = logsumexp(logweights, n);
    std::vector<double> pos, neg;
    for (int i = 0; i < n; ++i) {
        if (vals[i] > 0) pos.push_back(logweights[i] + log(vals[i]) - logsw);
        else if (vals[i] < 0) neg.push_back(logweights[i] + log(-vals[i]) - logsw);
    }
    if ((pos.size()>0) && (neg.size()>0)) {
        double pp = logsumexp(pos);
        double nn = logsumexp(neg);
        if (abs(pp-nn) < 1e-20) return 0.0;
        else if (pp-nn > 100) return exp(pp);
        else if (nn-pp > 100) return -exp(nn);
        else return exp(pp)*(1.0-exp(nn-pp));
    }
    else if (pos.size()>0) return exp(logsumexp(pos));
    else return -exp(logsumexp(neg));
}

