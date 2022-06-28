import os
import json
import pandas as pd
import pickle
import numpy as np
def write_sample_file(jobid, jobname, output_dir, loop, it):
  try:
      os.makedirs(output_dir)
  except FileExistsError:
      pass
  configuration={
    "logging": {
        "job_id": jobid,
        "job_name":  "{}_loop{}".format(jobname,loop),
        "logging_path": os.path.join(output_dir, "progress_t{}.log".format(it)),
        "recipient": "local",
        "sender": "http://127.0.0.1"
    },
    "parameters": {
        "model_path": os.path.join(output_dir, "results_t{}/Agent.ckpt".format(it)),
        "output_smiles_path": os.path.join(output_dir, "results_t{}/sampled.csv".format(it)),
        "num_smiles": 1024,
        "batch_size": 128,                          
        "with_likelihood": False
    },
    "run_type": "sampling",
    "model_type": "default",
    "version": 3
  }
  conf_filename = os.path.join(output_dir, "config_t{}_sampling.json".format(it))
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename

def write_config_file(jobid, jobname, reinvent_dir, reinvent_env, output_dir, fpdim, loop, it, modelfile, seed):
  # if required, generate a folder to store the results
  try:
      os.makedirs(output_dir)
  except FileExistsError:
      pass
  
  diversity_filter = {
    "name": "IdenticalMurckoScaffold",
    "bucket_size": 25,
    "minscore": 0.2,
    "minsimilarity": 0.4
  }

  inception = {
    "memory_size": 20,
    "sample_size": 5,
    "smiles": []
  }

  human_component = {
    "component_type": "predictive_property",
    "name": "Human-component",
    "weight": 1,
    "specific_parameters": {
      "model_path": modelfile,
      "gpflow": "regression",
      "descriptor_type": "ecfp",
      "size": fpdim,
      "container_type":"gpflow_container",
      "use_counts": True,
      "use_features": True,
      "transformation": {
        "transformation_type":"clipping",
        "low":0,
        "high":1
      }
    }
  }

  scoring_function = {
    "name": "custom_sum",
    "parallel": False,
    "parameters": [
      human_component
    ]
  }

  configuration = {
      "version": 3,
      "run_type": "reinforcement_learning",
      "model_type": "default",
      "parameters": {
          "scoring_function": scoring_function
      }
  }

  configuration["parameters"]["diversity_filter"] = diversity_filter
  configuration["parameters"]["inception"] = inception

  configuration["parameters"]["reinforcement_learning"] = {
      "prior": os.path.join(reinvent_dir, "data/random.prior.new"),
      "agent": os.path.join(reinvent_dir, "data/random.prior.new"),
      "n_steps": 300,
      "sigma": 128,
      "learning_rate": 0.0001,
      "batch_size": 128,
      "reset": 0,
      "reset_score_cutoff": 0.5,
      "margin_threshold": 50
  }

  configuration["logging"] = {
      "sender": "http://127.0.0.1",
      "recipient": "local",
      "logging_frequency": 0,
      "logging_path": os.path.join(output_dir, "progress_t{}.log".format(it)),
      "result_folder": os.path.join(output_dir, "results_t{}".format(it)),
      "job_name": "{}_loop{}".format(jobname,loop),
      "job_id": jobid
  }
  
  # write the configuration file to the disc
  conf_filename = os.path.join(output_dir, "config_t{}.json".format(it))
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  
  return conf_filename


def write_query_to_csv(smiles, ids, query, file, output_dir):
  try:
    os.mkdir(output_dir + '/query')
  except FileExistsError:
      pass
  data = {'id': [ids[i] for i in query], 'SMILES': smiles}
  df = pd.DataFrame(data)
  df.to_csv(file, index = False, header=True)


def write_run_sample(seed, output_dir, reinvent_env, reinvent_dir, step, n_iteration):
  runfile = output_dir + '/run_sampling.sh'
  array_num=int(np.ceil(n_iteration/step))
  try:
    os.mkdir(output_dir + '/slurm')
  except FileExistsError:
      pass
  with open(runfile, 'w') as f:
      f.write("#!/bin/bash -l \n")
      f.write("#SBATCH --mem=500M \n")
      f.write('#SBATCH --time=00:05:00 \n')
      f.write('#SBATCH -o {}/slurm/out_{}_sampling_%a.out\n'.format(output_dir, seed))
      f.write('#SBATCH --array=0-{}\n'.format(array_num))
      f.write('\n')
      f.write('module purge\n')
      f.write('module load anaconda\n')
      f.write('source activate {}\n'.format(reinvent_env))
      f.write('\n')
      f.write('config_index=$(($SLURM_ARRAY_TASK_ID*{}))\n'.format(step))
      f.write('conf_filename="{}/config_t${{config_index}}_sampling.json"\n'.format(output_dir))
      f.write('srun python {}/input.py $conf_filename\n'.format(reinvent_dir))

def write_runs_sh(seed, output_dir, reinvent_env, reinvent_dir, step, n_iteration):
  runfile = output_dir + '/runs.sh'
  array_num=int(np.ceil(n_iteration/step))
  try:
    os.mkdir(output_dir + '/slurm')
  except FileExistsError:
      pass
  with open(runfile, 'w') as f:
      f.write("#!/bin/bash -l \n")
      f.write("#SBATCH --mem=25G \n")
      f.write('#SBATCH --time=02:00:00\n')
      f.write('#SBATCH -o {}/slurm/out_{}_%a.out\n'.format(output_dir, seed))
      f.write('#SBATCH --array=0-{}\n'.format(array_num))
      f.write('\n')
      f.write('module purge\n')
      f.write('module load anaconda\n')
      f.write('source activate {}\n'.format(reinvent_env))
      f.write('\n')
      f.write('config_index=$(($SLURM_ARRAY_TASK_ID*{}))\n'.format(step))
      f.write('conf_filename="{}/config_t$config_index.json"\n'.format(output_dir))
      f.write('srun python {}/input.py $conf_filename\n'.format(reinvent_dir))

def write_idx(L0, U0, i_query, y_train, output_dir, loop, it):
  i_query=np.copy(i_query)
  y_response = y_train[i_query]
  i_generated_in_query=np.where(y_response==-1)[0]
  i_generated=i_query[i_generated_in_query] #indexes in i_query where points to generated data
  new_start_idx=np.sum(y_train!=-1) # the num of data with labels
  i_query[i_generated_in_query]= np.arange(new_start_idx,new_start_idx+len(i_generated))
  L0=np.union1d(L0,i_query)
  U0 = np.setdiff1d(U0,i_query)

  dat_save = {
    'L': L0, # indices of labeled data
    'U': U0,
  }
  try:
    os.mkdir(output_dir + '/idx')
  except FileExistsError:
      pass
  filename = output_dir + '/idx/log_loop{}_it{}.p'.format(loop,it)
  with open(filename , 'wb') as f:
      pickle.dump(dat_save, f)
      f.close()
  return L0, U0


def write_training_data(smiles, activity, id_train, output_dir, idx_query=None, num_original=None):

  
  if idx_query is not None:
    idx_original=np.arange(num_original,dtype='i')
    i_generated=idx_query[idx_query>=num_original]
    idx=np.append(idx_original,i_generated)
  else:
    idx=np.where(activity!=-1)[0]
  idx.astype(int)
  print('idx.shape is {}'.format(idx.shape))

  smiles, activity, smiles_id=smiles[idx], activity[idx], id_train[idx]
  dataset=pd.DataFrame({'id':smiles_id, 'canonical': smiles, 'activity': activity})
  dataset.to_csv(os.path.join(output_dir,'drd2.train.csv'),index=False)

