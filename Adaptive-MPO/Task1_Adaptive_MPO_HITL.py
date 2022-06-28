# load dependencies
import sys
import os
import re
import json
import tempfile
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import math
import reinvent_scoring
from numpy.random import default_rng

from scripts.lowSlogPQED import expert_qed, get_adsParameters
from scripts.write import write_REINVENT_config
from scripts.acquisition import select_query
from scripts.REINVENTconfig import parse_config_file

import pystan
import stan_utility
import pickle


def do_run(acquisition, seed):
    ##############################
    # Quick options
    FIT_MODEL = True # whether to fit a Stan model or not
    LOAD_MODEL = False # load Stan model from disc instead of fitting it
    SUBSAMPLE = True # Reduce the size of the pool of unlabeled molecules to reduce computation time
    ##############################

    jobid = 'demo_Task1'
    jobname = "Learn the parameters of MPO"
    np.random.seed(seed)
    rng = default_rng(seed)

    ########### HITL setup #################
    T = 10 # numer of HITL iterations (T in the paper)
    n = 10 # number of molecules shown to the simulated chemist at each iteration (n_batch in the paper)
    n0 = 10 # number of molecules shown to the expert at initialization (N_0 in the paper)
    K = 3 # number of REINVENT runs (K=R+1 in the paper): usage: K=2 for one HITL round (T*n+n0 queries); K=3 for two HITL rounds (2*(T*n+n0) queries)
    ########################################
    
    stanfile = 'usermodel_doublesigmoid'
    model_identifier = stanfile + '_' + str(jobid)

    # --------- change these path variables as required
    reinvent_dir = os.path.expanduser("/path/to/Reinvent")
    reinvent_env = os.path.expanduser("/path/to/conda_environment") 
    output_dir = os.path.expanduser("/path/to/result/directory/{}_seed{}".format(jobid,seed))
    print("Running MPO experiment with N0={}, T={}, R={}, n_batch={}, seed={}. \n Results will be saved at {}".format(n0, T, K-1, n, seed, output_dir))

    expert_score = []
    conf_filename='config.json'

    for REINVENT_iteration in np.arange(1,K):
        # if required, generate a folder to store the results
        READ_ONLY = False # if folder exists, do not overwrite results there
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            READ_ONLY = True
            print("Reading REINVENT results from file, no re-running.")
            pass

        if(not READ_ONLY):
            # write the configuration file to disc
            configuration_JSON_path = write_REINVENT_config(reinvent_dir, reinvent_env, output_dir, jobid, jobname)
            # run REINVENT
            os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + configuration_JSON_path + '&> ' + output_dir + '/run.err')
            
        with open(os.path.join(output_dir, "results/scaffold_memory.csv"), 'r') as file:
            data = pd.read_csv(file)
        
        colnames = list(data) 
        smiles = data['SMILES']
        total_score = data['total_score']
        high_scoring_threshold = 0.4
        high_scoring_idx = total_score >= high_scoring_threshold

        # Scoring component values
        scoring_component_names = [s.split("raw_")[1] for s in colnames if "raw_" in s]
        print("scoring components:")
        print(scoring_component_names)
        x = np.array(data[scoring_component_names])
        print('Scoring component matrix dimensions:')
        print(x.shape)

        # Only analyse highest scoring molecules
        x = x[high_scoring_idx,:]
        smiles = smiles[high_scoring_idx]
        total_score = total_score[high_scoring_idx]
        print('{} molecules'.format(len(smiles)))

        # Expert values (modified QED)
        s_mqed = np.zeros(len(smiles))
        for i in np.arange(len(smiles)):
            try:
                cur_mol = Chem.MolFromSmiles(smiles[i])
                s_mqed[i] = expert_qed(cur_mol)
            except:
                print("INVALID MOLECULE in scaffold memory")
                s_mqed[i] = 0
        expert_score += [s_mqed]
        print("Average modified QED in REINVENT output")
        print(np.mean(expert_score[REINVENT_iteration-1]))

        raw_scoring_component_names = ["raw_"+name for name in scoring_component_names] 
        x_raw = data[raw_scoring_component_names].to_numpy()
        x =  data[scoring_component_names].to_numpy()
        if(SUBSAMPLE):
            N0 = x_raw.shape[0]
            N = 10000 # Maximum number of molecules in U
            N = min(N0, N)
            sample_idx = rng.choice(N0, N, replace=False)
            x_raw = x_raw[sample_idx,:]
            x = x[sample_idx,:]
            smiles = smiles[sample_idx]
            try:
                user_score = expert_score[REINVENT_iteration-1][sample_idx]
            except IndexError:
                user_score = np.array([expert_qed(Chem.MolFromSmiles(si)) for si in smiles])
        else:
            try:
                user_score = expert_score[REINVENT_iteration-1]
            except IndexError:
                user_score = np.array([expert_qed(Chem.MolFromSmiles(si)) for si in smiles])
            
        N = x_raw.shape[0] # total number of molecules
        print("N_U={}".format(N))
        k = x_raw.shape[1]  # number of scoring functions
        w_e = np.ones(k)/k # equal weights

        # Generate simulated chemist's feedback
        y_all = rng.binomial(1, user_score) 
        # Select indices of feedback molecules at initialization (=iteration 0)
        selected_feedback = rng.choice(N, n0, replace=False)

        # Read desirability function (=score transformation) parameters from config file;
        # they will be used as prior means in the user-model
        conf0 = parse_config_file(os.path.join(output_dir, conf_filename), scoring_component_names)
        low0 = conf0['low']
        high0 = conf0['high']
        print("Prior means of low:")
        print(low0)
        print("Prior means of high:")
        print(high0)
        
        # fixed double sigmoid params from config file:
        coef_div = conf0['coef_div']
        coef_si = conf0['coef_si']
        coef_se = conf0['coef_se']
                
        # Read true values from a ground-truth config file
        gt_config = parse_config_file(os.path.join('./data/best_matching_params_modifiedQED_config.json'), scoring_component_names)
        low_true = gt_config['low']
        high_true = gt_config['high']

        mask = np.ones(N, dtype=bool)
        mask[selected_feedback] = False

        data_doublesigmoid = {
            'n': n0,                   
            'k': k,
            'x_raw': x_raw[selected_feedback,:],
            'y': y_all[selected_feedback],
            'weights': w_e,
            "coef_div": coef_div,
            "coef_si": coef_si,
            "coef_se": coef_se,
            'high0': high0,
            'low0': low0,
            'npred': N-len(selected_feedback),
            'xpred': x_raw[mask,:]
        }

        model_savefile = output_dir + '/{}_iteration_{}.pkl'.format(model_identifier, REINVENT_iteration-1)
        if(FIT_MODEL):
            print("compiling the Stan model")
            model = stan_utility.compile_model('./' + stanfile + '.stan', model_name=model_identifier)
            print("sampling")
            fit = model.sampling(data=data_doublesigmoid, seed=8453462, chains=4, iter=1000, n_jobs=1)
            print("Saving the fitted model to {}".format(model_savefile))
            pickle.dump({'model': model, 'fit': fit}, open(model_savefile, 'wb' ), protocol=-1)
        if(LOAD_MODEL):
            print("Loading the fit")
            data_dict = pickle.load(open(model_savefile, 'rb'))
            fit = data_dict['fit']
            model = data_dict['model']
        la = fit.extract(permuted=True)  # return a dictionary of arrays for each model parameter
        # compute errors in learned limits
        low = np.mean(la['lows'],axis=0)
        high = np.mean(la['highs'],axis=0)
        parameter_order = ['low{}'.format(i) for i in np.arange(len(low0))] + ['high{}'.format(i) for i in np.arange(len(high0))]
        thetas = np.hstack((low,high))
        thetas_true = np.hstack((low_true, high_true))
        errs = [(thetas_true - thetas)**2] # MSE
        mean_limits =[thetas]

        # Diagnostic tests
        stan_utility.check_all_diagnostics(fit)
        warning0 = stan_utility.check_all_diagnostics(fit, quiet=True)

        print("highs")
        for i in np.arange(7):
            print(high[i])
        print("lows")
        for i in np.arange(7):
            print(low[i])

        y = y_all[selected_feedback]
        n_accept = [sum(y)] # number of accepted molecules at each iteration
        warning_code = [warning0]

        ########################### HITL rounds ######################################
        for t in np.arange(T):
            print("iteration t={}".format(t))
            # query selection
            new_query = select_query(N, n, fit, selected_feedback, acquisition, rng)
            # get simulated chemist's responses
            new_y = rng.binomial(1, user_score[new_query])
            n_accept += [sum(new_y)]
            print("Feedback idx at iteration {}:".format(t))
            print(new_query)
            print("Number of accepted molecules at iteration {}: {}".format(t,n_accept[t]))
            # append feedback
            selected_feedback = np.hstack((selected_feedback, new_query))
            y = np.hstack((y, new_y))
            mask = np.ones(N, dtype=bool)
            mask[selected_feedback] = False
            data_doublesigmoid = {
                'n': len(selected_feedback),                   
                'k': k,
                'x_raw': x_raw[selected_feedback,:],
                'y': y,
                'weights': w_e,
                "coef_div": coef_div,
                "coef_si": coef_si,
                "coef_se": coef_se,
                'high0': high0,
                'low0': low0,
                'npred': N-len(selected_feedback),
                'xpred': x_raw[mask,:]
            }
            # re-fit model
            fit = model.sampling(data=data_doublesigmoid, seed=8453462, chains=4, iter=1000, n_jobs=1)
            stan_utility.check_all_diagnostics(fit)
            code = stan_utility.check_all_diagnostics(fit, quiet=True)
            warning_code += [code]
            la = fit.extract(permuted=True)
            low = np.mean(la['lows'],axis=0)
            high = np.mean(la['highs'],axis=0)
            thetas = np.hstack((low,high))
            errs += [(thetas_true - thetas)**2]
            mean_limits += [thetas]
        
        # Posterior mean of parameters
        lows = la['lows']
        highs = la['highs']
        m_high = np.mean(highs, axis=0)
        m_low = np.mean(lows, axis=0)
        
        x = np.arange(T+1)
        true = np.hstack((low_true, high_true))
        rerrs = np.absolute(mean_limits - true) / np.absolute(true)
        plt.plot(x, rerrs)
        plt.ylabel("relative error: |error|/true")
        plt.xlabel("Number of iterations")
        plt.legend(parameter_order)
        plt.title("Relative errors in learned limits")
        plt.savefig(os.path.join(output_dir, '{}_relative_abs_error_{}.png'.format(jobid, acquisition)), bbox_inches='tight')
        plt.close()

        plt.plot(x, np.mean(rerrs, axis=1))
        plt.title("Mean relative error in learned limits")
        plt.ylabel("relative error: |error|/true")
        plt.xlabel("Number of iterations")
        plt.savefig(os.path.join(output_dir, '{}_relative_abs_error_mean_{}.png'.format(jobid, acquisition)), bbox_inches='tight')
        plt.close()

        #### SAVE  RESULTS ###
        dat_save = {'mean limits': mean_limits, 'true limits': true, 'rerrs': rerrs}
        filename = output_dir + '/log_{}_it{}.p'.format(acquisition,T)
        with open(filename , 'wb') as f:
            pickle.dump(dat_save, f)

        print("Check convergence diagnostics of Stan: bits from right to left: n_eff, r_hat, div, treedepth, energy")
        for t in np.arange(len(warning_code)):
            print("t={}".format(t))
            print(bin(warning_code[t]))


        # Define directory for the next round
        output_dir_iter = os.path.join(output_dir, "iteration{}_{}".format(REINVENT_iteration, acquisition))
        READ_ONLY = False
        # if required, generate a folder to store the results
        try:
            os.mkdir(output_dir_iter)
        except FileExistsError:
            READ_ONLY = True
            print("Reading REINVENT results from file, no re-running.")
            pass

        def set_scoring_component_parameters(configuration, params):
            # modify data structure for easy access to components by their name
            scc = {}
            for comp in configuration["parameters"]["scoring_function"]["parameters"]:
                scc[comp["name"]] = comp
            
            for name, p in params.items():
                for key, value in p.items():
                    print("Writing component " + name + ": " + key + "=" + str(value))
                    scc[name]["specific_parameters"]["transformation"][key] = value
        
        # modify parameters of the score transformations    
        configuration = json.load(open(os.path.join(output_dir, conf_filename)))
        params = {}
        for i, comp in enumerate(scoring_component_names):
            params[comp] = {'high': m_high[i], 'low': m_low[i]}
        set_scoring_component_parameters(configuration, params)
        print(configuration)

        # modify log and result paths in configuration
        configuration["logging"]["logging_path"] = os.path.join(output_dir_iter, "progress.log")
        configuration["logging"]["result_folder"] = os.path.join(output_dir_iter, "results")

        if(not READ_ONLY):
            conf_filename = "iteration{}_config.json".format(REINVENT_iteration)
            configuration_JSON_path = os.path.join(output_dir_iter, conf_filename)
            # write the updated configuration file to the disc
            with open(configuration_JSON_path, 'w') as f:
                json.dump(configuration, f, indent=4, sort_keys=True)

        # Run REINVENT again
        if(not READ_ONLY):
            os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + configuration_JSON_path + '&> ' + output_dir_iter + '/run.err')

        with open(os.path.join(output_dir_iter, "results/scaffold_memory.csv"), 'r') as file:
            data_it1 = pd.read_csv(file)

        # Last round: analyze results 
        if REINVENT_iteration == K-1:
            # extract SMILES from scaffold memory 
            smiles_it1 = data_it1['SMILES']
            total_score_it1 = data_it1['total_score']
            high_scoring_idx_it1 = total_score_it1 >= high_scoring_threshold

            scoring_component_names = [s.split("raw_")[1] for s in colnames if "raw_" in s]
            x_it1 = np.array(data_it1[scoring_component_names])

            # Only analyse highest scoring molecules
            x_it1 = x_it1[high_scoring_idx_it1,:]
            smiles_it1 = smiles_it1[high_scoring_idx_it1]
            total_score_it1 = total_score_it1[high_scoring_idx_it1]
            print('{} molecules'.format(len(smiles_it1)))

            # Expert values (modified QED)
            s_mqed = np.zeros(len(smiles_it1))
            for i in np.arange(len(smiles_it1)):
                try:
                    cur_mol = Chem.MolFromSmiles(smiles_it1[i])
                    s_mqed[i] = expert_qed(cur_mol)
                except:
                    s_mqed[i] = 0
            expert_score += [s_mqed]

        for i in np.arange(len(expert_score)):
            print("Iteration " + str(i))
            print("Average modified QED in REINVENT output")
            print(np.mean(expert_score[i]))
            print("Number of molecules with modified QED score > 0.8")
            print(np.sum([int(sc >= 0.8) for sc in expert_score[i]]))
            print("Number of molecules with modified QED score > 0.9")
            print(np.sum([int(sc >= 0.9) for sc in expert_score[i]]))

        dat_save = {'mean limits': mean_limits, 'true limits': true, 'rerrs': rerrs, 'expert_score': expert_score}
        filename = output_dir + '/log_{}_it{}.p'.format(acquisition,T)
        with open(filename , 'wb') as f:
            pickle.dump(dat_save, f)
        
        # Set output dir and configuration file name of the next round:
        output_dir = output_dir_iter
        conf_filename = "iteration{}_config.json".format(REINVENT_iteration)

    # Plot the final result
    r = np.arange(len(expert_score))
    m_score = [np.mean(expert_score[i]) for i in r]
    plt.plot(r, m_score)
    plt.title("Performance of {}".format(acquisition))
    plt.ylabel("Average of modified QED score in REINVENT output")
    plt.xlabel("Rounds")
    plt.savefig(os.path.join(output_dir, '{}_mQED_{}.png'.format(jobid, acquisition)), bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    print(sys.argv)
    acquisition = sys.argv[1] # acquisition: 'uncertainty', 'random', 'thompson', 'greedy'
    seed = int(sys.argv[2])
    do_run(acquisition, seed)