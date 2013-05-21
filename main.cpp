//
//  main.cpp
//  GaussianMarkovNetworks
//
//  Created by Rongjing Xiang on 2/29/12.
//  Copyright 2012 Purdue University. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "network.h"
#include "clmn.h"
#include "lgmn.h"
#include "boost/scoped_array.hpp"

DEFINE_string(dataset_path, "/Users/rxiang/Projects/clmn/script/amazonmusic/music", "path to network data");
DEFINE_string(identifier, "", "identifier of experiment");
DEFINE_int32(model, 2, "0: use CLMN; 1: use LR; 2: use LGMN");

using namespace std;

int main (int argc, char * argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    boost::scoped_ptr<Network> data (new Network);
    data->load_data_from_file(FLAGS_dataset_path);
    string pref = FLAGS_dataset_path + fLS::FLAGS_identifier;
    ifstream fin((pref + "_labeled.txt").c_str());
    string fn;
    if (FLAGS_model == 0) fn = pref + "_clmn_pred.txt";
    else if (FLAGS_model == 1) fn = pref + "_lr_pred.txt";
    else fn = pref + "_lgmn_pred.txt";
    ofstream fout(fn.c_str());
    if (FLAGS_model <= 1) {
    CopulaLatentMarkovNet clmn(data.get());
    int c = 0;
    while (true) {
        int nlabeled;
        fin >> nlabeled;
        if (nlabeled == -1) break;
        LOG(INFO) << "Case " << c << ": # of labeled: " << nlabeled << ", total: " << data->num_instances_;
        vector<int> labeled(nlabeled);
        for (int i = 0; i < nlabeled; ++i) {
            int node;
            fin >> node;
            labeled[i] = node;
        }
        
        clmn.set_labeled(labeled);
        vector<double> preds;
        if (FLAGS_model == 0) {
            clmn.expectation_propagation();
            clmn.predict(preds);
        } else clmn.lr_predict(preds);
        CHECK(preds.size() == data->num_instances_-nlabeled) << preds.size()  << ' ' << data->num_instances_-nlabeled;
        for (int i = 0; i < preds.size(); ++i) {
            if (i > 0) fout << " ";
            fout << preds[i];
        }
        fout << endl;
        ++c;
//        break;
    }
    } else { // apply LGMN
        LatentGaussMarkovNet lgmn(data.get());
        int c = 0;
        while (true) {
            int nlabeled;
            fin >> nlabeled;
            if (nlabeled == -1) break;
            LOG(INFO) << "Case " << c << ": # of labeled: " << nlabeled << ", total: " << data->num_instances_;
            vector<int> labeled(nlabeled);
            for (int i = 0; i < nlabeled; ++i) {
                int node;
                fin >> node;
                labeled[i] = node;
            }
            
            lgmn.set_labeled(labeled);
            vector<double> preds;
            lgmn.expectation_propagation();
            lgmn.predict(preds);
            CHECK(preds.size() == data->num_instances_-nlabeled) << preds.size()  << ' ' << data->num_instances_-nlabeled;
            for (int i = 0; i < preds.size(); ++i) {
                if (i > 0) fout << " ";
                fout << preds[i];
            }
            fout << endl;
            ++c;
        }
    }
    fin.close();
    fout.close();
}

