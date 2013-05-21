//
//  lgmn.cpp
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 3/20/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#include "lgmn.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/math/distributions/normal.hpp"

#define PI 3.141592653589793
DECLARE_int32(num_ep_iters);
DECLARE_double(ep_dampening);
DECLARE_int32(ep_randseed);
DECLARE_int32(num_marginal_samples);
DECLARE_double(laplacian_epsilon);
DECLARE_int32(num_mh_samples);
DECLARE_int32(min_mh_samples);
DECLARE_double(epsilon_mh);

#define sqrt2   1.4142135623730950488
#define pi2     6.2831853071795864770 
#define sqrt2pi 2.5066282746310005024

double erf(double x){    
    // constants    
    double a1 =  0.254829592;    
    double a2 = -0.284496736;    
    double a3 =  1.421413741;    
    double a4 = -1.453152027;    
    double a5 =  1.061405429;    
    double p  =  0.3275911;    
    // Save the sign of x    
    int sign = 1;    
    if (x < 0)        sign = -1;    
    x = fabs(x);    // A&S formula 7.1.26    
    double t = 1.0/(1.0 + p*x);    
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);    
    return sign*y;
}

/* GAUSSIAN DENSITY FUNCTION. */
double gsl_ran_ugaussian_pdf(double x)
{
    return exp(-x*x/2)/sqrt2pi ;
}

double gsl_cdf_ugaussian_P(double x)
{
    return 0.5+0.5*erf(x/sqrt2) ;
}

double logphi(double x)
{
    double phi = gsl_cdf_ugaussian_P(x);
    const double xmin = -6.2;
    const double xmax = -5.5;
    if (x>xmax)
        return log(phi);
    else { 
        double lam = 1.0/(1.0+exp(25*(0.5-(x-xmin)/(xmax-xmin))));
        double logphi = -0.57236494292470 - x*x*0.5 - log( sqrt(x*x*0.5+2) - x*0.70710678118655 );
        if (x>xmin) // interp.
            logphi = (1-lam)*logphi + lam*log(phi); 
        return logphi; 
    }
}

double GaussOverPhi(double x)
{
    double phi = exp(logphi(x));
    const double xmax = -5;
    const double xmin = -6;
    if (x>xmax)
        return gsl_ran_ugaussian_pdf(x)/phi;
    else {
        double diff = sqrt(x*x*0.25+1) - x*0.5;
        if (x<xmin) // difficult
            return diff;
        else { // interp.
            double lam = -5-x;
            return (1-lam)*(gsl_ran_ugaussian_pdf(x)/phi)+lam*diff;
        }
    }
}

LatentGaussMarkovNet::LatentGaussMarkovNet(Network* data) : data_(data) {
    num_nodes_ = data_->num_instances_;
    SparseMatrix<double, RowMajor> laplacian = data_->get_graph_laplacian();
    messages_.reset(new map<int, message>[num_nodes_]);
    laplacian_diag_.reset(new double[num_nodes_]);
    for (int i = 0; i < num_nodes_; ++i) {
        VectorXd d(num_nodes_);
        for (int j = 0; j < num_nodes_; ++j) d(j) = 0.0;
        d(i) = 1.0;
        map<int, message>* linked = &(messages_[i]);
        for (SparseMatrix<double, RowMajor>::InnerIterator it(laplacian,i); it; ++it) 
            if (it.col() == i) laplacian_diag_[i] = it.value() - fLD::FLAGS_laplacian_epsilon + 1e-3;
            else {
                message msg;
                msg.val = it.value();
                msg.iv = 0.0;
                msg.h = 0.0;
                (*linked)[it.col()] = msg;
            }
    }
}

void LatentGaussMarkovNet::set_labeled(const vector<int>& labeled) {
    observed_labels_.reset(new int[num_nodes_]);
    for (int i = 0; i < num_nodes_; ++i) observed_labels_[i] = 0;
    for (vector<int>::const_iterator it = labeled.begin(); it != labeled.end(); ++it) {
        int id = *it;
        observed_labels_[id] = data_->get_instance_label(id);
    }
}

pair<double, double> LatentGaussMarkovNet::update_by_likelihood(int node, double s_iv, double s_h, double m_all, double v_all) {
    double m = s_h/s_iv;
    double v = 1.0/s_iv;
    double y = observed_labels_[node];
    
    /*
    boost::scoped_array<double> logweights(new double[FLAGS_num_mh_samples]);
    boost::scoped_array<double> t(new double[FLAGS_num_mh_samples]);    
    boost::math::normal qnd(m, sqrt(v));    
    for (int sample = 0; sample < FLAGS_num_mh_samples; ++sample) {
        double u = random_->Random();
        double g = quantile(qnd, u);
        t[sample] = g;
        double s = -y*g;
//        double loggaussdif = 0.5*((g-m_all)*(g-m_all)/v_all - (g-m)*(g-m)/v);
        if (s > 20) logweights[sample] = -s;
        else logweights[sample] = - log(1.0+exp(s));
    }     
    double mnew = weighted_average(logweights.get(), t.get(), FLAGS_num_mh_samples);
    for (int sample = 0; sample < FLAGS_num_mh_samples; ++sample) t[sample] *= t[sample];
    double vnew = weighted_average(logweights.get(), t.get(), FLAGS_num_mh_samples) - mnew*mnew;    
    double den = 1.0 + (PI/8) * v;
    double sqrtden = sqrt(den);
    double z = y*m/sqrtden;
    double alpha = 1.0/(1.0+exp(z))/sqrtden;
    double beta = alpha*(alpha+(PI/8)*y*m)/den;
    double mnew = m + v*y*alpha;
    double vnew = v-v*v*beta;
     */
    double z = y*m/sqrt(1.0+v);
    double alpha = 1.0/sqrt(1.0+v)*GaussOverPhi(z);
    double beta = alpha*(alpha+y*m/(1.0+v));
    double mnew = m + v*y*alpha;
    double vnew = v-v*v*beta;
    /*
    double epsilon = FLAGS_epsilon_mh;
    double epsilon2 = sqrt(1.0-epsilon*epsilon);
    double mnew = 0.0;
    double vnew = 0.0;
    double nacc = 0.0;
    int count = 0;
    double oldlikelihood = -1e100;
    boost::math::normal qnd(m_all, sqrt(v_all));
    double oldt;
    while (nacc < FLAGS_num_mh_samples) {
        double u = random_->Random();
        double g = quantile(qnd, u);
        double newt = g;
        if (count > 0) newt = epsilon*g + epsilon2*oldt;
        double newlikelihood = 0.5*((newt-m_all)*(newt-m_all)/v_all - (newt-m)*(newt-m)/v);
        double s = -y*newt;
        if (s>20) newlikelihood += -s;
        else newlikelihood += -log(1.0+exp(s));
        if (log(random_->Random()) < newlikelihood-oldlikelihood) {
            oldt = newt;
            mnew += newt;
            vnew += newt*newt;
            nacc += 1;
            oldlikelihood = newlikelihood;
        }
        count += 1;
        if ((count > 100) and (nacc < 0.1*count)) {
            epsilon = epsilon*0.8;
            epsilon2 = sqrt(1.0-epsilon*epsilon);
            LOG(WARNING) << "acceptance rate < 10%, restart markov chain with epsilon " << epsilon;
            if (epsilon < 1e-10) {
                LOG(WARNING) << "epsilon < 1e-10, " << nacc << " samples obtained, giving up";
                return make_pair(0.0, 0.0);
            }
            oldlikelihood = -1e100;
            nacc = 0.0;
            count = 0;
            mnew = 0.0;
            vnew = 0.0;
        }
    }
    mnew /= nacc;
    vnew = vnew/nacc - mnew*mnew;
     */
    return make_pair(mnew, vnew);
}

void LatentGaussMarkovNet::expectation_propagation() {
    random_.reset(new CRandomMersenne(FLAGS_ep_randseed));
    // initialize messages
    m_by_ep_.reset(new double[num_nodes_]);
    v_by_ep_.reset(new double[num_nodes_]);
    for (int i = 0; i < num_nodes_; ++i) {
        m_by_ep_[i] = 0.0;
        v_by_ep_[i] = 1.0;
        for (map<int, message>::iterator it = messages_[i].begin(); it != messages_[i].end(); ++it) {
            (it->second).iv = 0.0;
            (it->second).h = 0.0;
        }
    }
    double min_dif = 1e10;
    for (int iter = 0; iter < FLAGS_num_ep_iters * 2; ++iter) {
        double step_size = exp(-(double)FLAGS_ep_dampening*iter);
        int nskipped2 = 0, nskipped = 0, nskipped3 = 0;
        double dif = 0.0;
        for (int i = 0; i < num_nodes_; ++i) {
            //            LOG(INFO) << "processing node " << i;
            double s_iv = laplacian_diag_[i];
            double s_h = 0.0;
            for (map<int, message>::const_iterator it = messages_[i].begin(); it != messages_[i].end(); ++it) {
                message msg(messages_[it->first][i]);
                s_iv += msg.iv;
                s_h += msg.h;
            }
            if (s_iv < 1e-20) {
                nskipped += 1;
                continue;
            }
            if (observed_labels_[i] == 0) { //unlabeled nodes
                m_by_ep_[i] = s_h/s_iv;
                v_by_ep_[i] = 1.0/s_iv;
            } else {                
                double m0 = m_by_ep_[i];
                double v0 = v_by_ep_[i];
                if (iter == 0) {
                    m0 = s_h/s_iv;
                    v0 = 1.0/s_iv;
                }
                pair<double, double> p(update_by_likelihood(i, s_iv, s_h, m0, v0));
                //            LOG(INFO) << "finish mh";
                if (p.second < 1e-20) {
                    nskipped3 += 1;
                    continue;
                }                
                s_iv = 1.0/p.second;
                s_h = p.first/p.second;
                m_by_ep_[i] = p.first;
                v_by_ep_[i] = p.second;
            }
            //            LOG(INFO) << "start message passing...";
            for (map<int, message>::iterator it = messages_[i].begin(); it != messages_[i].end(); ++it) {
                double old_iv = it->second.iv;
                double old_h = it->second.h;
                double val = it->second.val;
                message msg_jtoi(messages_[it->first][i]);
                double iv_inotj = s_iv - msg_jtoi.iv;
                if (abs(iv_inotj) < 1e-20) {
                    nskipped2 += 1;
                    continue;
                }
                double new_iv = -val*val/iv_inotj;
                double new_h = -val*(s_h-msg_jtoi.h)/iv_inotj;
                dif += abs(new_h-old_h);
                it->second.iv = old_iv * (1.0-step_size) + new_iv * step_size;
                it->second.h = old_h * (1.0-step_size) + new_h * step_size;
            }
            //            LOG(INFO) << "finish message passing...";
        }
        LOG(INFO) << "EP iter " << iter << ", message difference: " << dif << "# skipped: " << nskipped << " " << nskipped2 << " " << nskipped3;
        if ((iter > FLAGS_num_ep_iters/5) && (dif < min_dif)) min_dif = dif;
        if ((iter >= FLAGS_num_ep_iters) && (dif < min_dif*10)) break;
    }
}

void LatentGaussMarkovNet::predict(vector<double>& prediction) {
    random_.reset(new CRandomMersenne(FLAGS_ep_randseed));    
    for (int i = 0; i < num_nodes_; ++i) if (observed_labels_[i] == 0) {
        //        LOG(INFO) << "node " << i << ": " << m_by_ep_[i] << " " << v_by_ep_[i];
        
        double prob = 0.0;
        boost::math::normal qnd(m_by_ep_[i], sqrt(v_by_ep_[i]));
        for (int sample = 0; sample < FLAGS_num_marginal_samples; ++sample) {
            double u = random_->Random();
            //            LOG(INFO) << "sample " << sample << ": " << "u: ";
            double t = quantile(qnd, u);
//            prob += 1.0/(1.0+exp(-t)); // logistic
            prob += gsl_cdf_ugaussian_P(t); // probit
        }
         
        prediction.push_back(gsl_cdf_ugaussian_P(m_by_ep_[i]/sqrt(1.0+v_by_ep_[i])));
//       double z = m_by_ep_[i]/sqrt(1.0+PI/8*v_by_ep_[i]);
//        prediction.push_back(1.0/(1.0+exp(-z)));
       // prediction.push_back(m_by_ep_[i]);
    }
}
