//
//  clmn.cpp
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 3/1/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#include "clmn.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "unsupported/Eigen/CholmodSupport"
#include "boost/math/distributions/logistic.hpp"

DEFINE_double(epsilon_mh, 0.9, "starting step size of metropolis-hasting in EP");
DEFINE_int32(num_mh_samples, 200, "number of samples in numerical integration for EP");
DEFINE_int32(num_marginal_samples, 50, "number of samples in numerical integration for marginal prediction");
DEFINE_int32(min_mh_samples, 200, "minimum number of accepted samples needed in Metropolis-Hasting");
DEFINE_int32(num_ep_iters, 10, "number of EP iterations");
DEFINE_double(ep_dampening, 0.125, "EP dampening factor");
DEFINE_int32(ep_randseed, 1, "random seed used in EP");

CopulaLatentMarkovNet::CopulaLatentMarkovNet(Network* data) : data_(data) {
    num_nodes_ = data_->num_instances_;
    SparseMatrix<double, RowMajor> laplacian = data_->get_graph_laplacian();
    CholmodDecomposition<SparseMatrix<double, RowMajor> > solver;
    solver.compute(laplacian);
    CHECK(solver.info() == Eigen::Success);
    gmn_marginal_scale_.reset(new double[num_nodes_]);
    messages_.reset(new map<int, message>[num_nodes_]);
    laplacian_diag_.reset(new double[num_nodes_]);
    for (int i = 0; i < num_nodes_; ++i) {
        VectorXd d(num_nodes_);
        for (int j = 0; j < num_nodes_; ++j) d(j) = 0.0;
        d(i) = 1.0;
        VectorXd x = solver.solve(d);
        CHECK(solver.info() == Eigen::Success);        
        gmn_marginal_scale_[i] = sqrt(x(i));
        map<int, message>* linked = &(messages_[i]);
        for (SparseMatrix<double, RowMajor>::InnerIterator it(laplacian,i); it; ++it) 
            if (it.col() == i) laplacian_diag_[i] = it.value();
            else {
                message msg;
                msg.val = it.value();
                msg.iv = 0.0;
                msg.h = 0.0;
                (*linked)[it.col()] = msg;
            }
    }
}

void CopulaLatentMarkovNet::set_labeled(const vector<int>& labeled) {
    logistic_.reset(new Logistic(data_->feature_dim_));
    observed_labels_.reset(new int[num_nodes_]);
    for (int i = 0; i < num_nodes_; ++i) observed_labels_[i] = 0;
    for (vector<int>::const_iterator it = labeled.begin(); it != labeled.end(); ++it) {
        int id = *it;
        observed_labels_[id] = data_->get_instance_label(id);
        logistic_->add_instance(data_->get_instance_attributes(id), observed_labels_[id]);
    }
    logistic_->train();
    logistic_->clear_data();
    cut_.reset(new double[num_nodes_]);
    for (int i = 0; i < num_nodes_; ++i) {
        logistic_->add_instance(data_->get_instance_attributes(i), data_->get_instance_label(i));
        if (observed_labels_[i] != 0) {
            boost::math::logistic ld(logistic_->get_mean(i), logistic_->scale_);
            cut_[i] = cdf(ld, 0.0);            
        }
    }
}

pair<double, double> CopulaLatentMarkovNet::metropolis_hasting(int node, double s_iv, double s_h) {
    double m = s_h/s_iv;
    double msigma = gmn_marginal_scale_[node];
    boost::math::normal nd(0.0, msigma);
    double c2 = -0.5*(s_iv - 1.0/(msigma*msigma));
    double mnew;
    double vnew;
    if (observed_labels_[node] != 0) { // labeled node, sample from the marginals
        double lower = 0.0;
        double upper = 1.0;
        if (observed_labels_[node] == -1) upper = cut_[node];
        else lower = cut_[node];
        boost::scoped_array<double> t(new double[FLAGS_num_mh_samples]);
        boost::scoped_array<double> logweights(new double[FLAGS_num_mh_samples]);
        for (int sample = 0; sample < FLAGS_num_mh_samples; ++sample) {
            double u = random_->Random();
            u = u * (upper - lower) + lower;
            double ts = quantile(nd, u);
            t[sample] = ts;
            logweights[sample] = c2*ts*ts + s_h*ts;
        }
        mnew = weighted_average(logweights.get(), t.get(), FLAGS_num_mh_samples);
        for (int sample = 0; sample < FLAGS_num_mh_samples; ++sample) t[sample] *= t[sample];
        vnew = weighted_average(logweights.get(), t.get(), FLAGS_num_mh_samples) - mnew*mnew;
    } else { // unlabeled node, sample from the estimated Gaussian
        double epsilon = FLAGS_epsilon_mh;
        double epsilon2 = sqrt(1.0-epsilon*epsilon);
        mnew = 0.0;
        vnew = 0.0;
        double nacc = 0.0;
        int count = 0;
        double oldlikelihood = -1e100;
        boost::math::normal qnd(m, 1.0/sqrt(s_iv));
        double oldt;
        while (nacc < FLAGS_min_mh_samples) {
            double u = random_->Random();
            double g = quantile(qnd, u);
            double newt = g;
            if (count > 0) newt = epsilon*g + epsilon2*oldt;
            double newlikelihood = compute_loglikelihoodpart(newt, nd, msigma);
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
//                LOG(WARNING) << "acceptance rate < 10%, restart markov chain with epsilon " << epsilon;
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
    }
    return make_pair(mnew, vnew);
}
    
double CopulaLatentMarkovNet::compute_loglikelihoodpart(double ts, const boost::math::normal& nd, double msigma) {
    // make use the quantile function of logistic distribution
    double ss = logistic_->scale_;
    double logcdf = log(cdf(nd, ts));
    double logsf = log(cdf(complement(nd, ts)));
    double lres = -logcdf;
    /*
    if (abs(logcdf-logsf) < 50) lres = log(1.0+exp(logsf-logcdf));
    else if (logsf > logcdf) lres = logsf;
    else lres = -logcdf;
     */
    double logisticlog = logsf - logcdf - log(ss) - 2.0*lres;
    double gausslog = ts/msigma;
    gausslog = -0.5*gausslog*gausslog;
    return logisticlog - gausslog;
}

void CopulaLatentMarkovNet::expectation_propagation() {
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
//            LOG(INFO) << "start mh..";
            pair<double, double> p(metropolis_hasting(i, s_iv, s_h));
//            LOG(INFO) << "finish mh";
            if (p.second < 1e-20) {
                nskipped3 += 1;
                continue;
            }
            s_iv = 1.0/p.second;
            s_h = p.first/p.second;
            m_by_ep_[i] = p.first;
            v_by_ep_[i] = p.second;
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

void CopulaLatentMarkovNet::lr_predict(vector<double>& prediction) {
    for (int i = 0; i < num_nodes_; ++i) if (observed_labels_[i] == 0) 
        prediction.push_back(logistic_->get_mean(i));
}

void CopulaLatentMarkovNet::predict(vector<double>& prediction) {
    random_.reset(new CRandomMersenne(FLAGS_ep_randseed));
    for (int i = 0; i < num_nodes_; ++i) if (observed_labels_[i] == 0) {
//        LOG(INFO) << "node " << i << ": " << m_by_ep_[i] << " " << v_by_ep_[i];
        boost::math::normal nd(0.0, gmn_marginal_scale_[i]);
        boost::math::logistic ld(logistic_->get_mean(i), logistic_->scale_);
        double z = 0.0;
        for (int sample = 0; sample < FLAGS_num_marginal_samples; ++sample) {
            double u = random_->Random();
//            LOG(INFO) << "sample " << sample << ": " << "u: ";
            boost::math::normal qnd(m_by_ep_[i], v_by_ep_[i]);
            double t = quantile(qnd, u);
//            LOG(INFO) << "t: " << t;
            u = cdf(nd, t);
//            LOG(INFO) << "u: " << u;
            if (u > 1.0-1e-10) u = 1.0 - 1e-10;
            else if (u < 1e-10) u = 1e-10;
            z += quantile(ld, u);
        }
        prediction.push_back(z/FLAGS_num_marginal_samples);
         /*
        double u = cdf(nd, m_by_ep_[i]);
        if (u > 1.0-1e-10) u = 1.0 - 1e-10;
        else if (u < 1e-10) u = 1e-10;
        prediction.push_back(quantile(ld, u));
         */
    }
}
