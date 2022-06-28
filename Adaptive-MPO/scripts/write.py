# Scripts for writing and modifying configuration json files of REINVENT

import os
import json

def write_REINVENT_config(reinvent_dir, reinvent_env, output_dir, jobid, jobname):

    diversity_filter = {
    "name": "IdenticalMurckoScaffold",
    "bucket_size": 25,
    "minscore": 0.4,
    "minsimilarity": 0.4
    }

    inception = {
    "memory_size": 20,
    "sample_size": 5,
    "smiles": []
    }

    component_mv = {
    "component_type": "molecular_weight",
    "name": "Molecular weight",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 700,
            "low": 50,
            "coef_div": 175.77,
            "coef_si": 2,
            "coef_se": 2
        }
    }
    }

    component_slogp = {    
    "component_type": "slogp",
    "name": "SlogP",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 10,
            "low": 3,
            "coef_div": 3.0,
            "coef_si": 2,
            "coef_se": 2
        }
    }
    }

    component_hba = {
    "component_type": "num_hba_lipinski",
    "name": "HB-acceptors (Lipinski)",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 11,
            "low": 2,
            "coef_div": 4.42,
            "coef_si": 2,
            "coef_se": 4.4
        }
    }
    }

    component_hbd = {
    "component_type": "num_hbd_lipinski",
    "name": "HB-donors (Lipinski)",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 8,
            "low": 1,
            "coef_div": 2.41,
            "coef_si": 2,
            "coef_se": 2
        }
    }
    }

    component_psa = {
    "component_type": "tpsa",
    "name": "PSA",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 300,
            "low": 100,
            "coef_div": 75.34,
            "coef_si": 2,
            "coef_se": 2
        }
    }
    }

    component_rotatable_bonds = {
    "component_type": "num_rotatable_bonds",
    "name": "Number of rotatable bonds",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 20,
            "low": 5,
            "coef_div": 5.69,
            "coef_si": 2,
            "coef_se": 2
        }
    }
    }

    component_num_rings = {
    "component_type": "num_rings",
    "name": "Number of aromatic rings",
    "weight": 1,
    "specific_parameters": {
        "transformation": {
            "transformation_type": "double_sigmoid",
            "high": 10,
            "low": 1,
            "coef_div": 2.28,
            "coef_si": 2,
            "coef_se": 2
        }
    }
    }

    scoring_function = {
    "name": "custom_product",
    "parallel": True,
    "parameters": [
        component_mv,
        component_slogp,
        component_hbd,
        component_hba,
        component_psa,
        component_rotatable_bonds,
        component_num_rings
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
        "logging_path": os.path.join(output_dir, "progress.log"),
        "result_folder": os.path.join(output_dir, "results"),
        "job_name": jobname,
        "job_id": jobid
    }

    # write the configuration file to disc
    configuration_JSON_path = os.path.join(output_dir, "config.json")
    with open(configuration_JSON_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=True)
    
    return configuration_JSON_path


def write_sample_file(jobid, jobname, agent_dir, N):
  configuration={
    "logging": {
        "job_id": jobid,
        "job_name":  "sample_agent_{}".format(jobname),
        "logging_path": os.path.join(agent_dir, "sampling.log"),
        "recipient": "local",
        "sender": "http://127.0.0.1"
    },
    "parameters": {
        "model_path": os.path.join(agent_dir, "Agent.ckpt"),
        "output_smiles_path": os.path.join(agent_dir, "sampled_N_{}.csv".format(N)),
        "num_smiles": N,
        "batch_size": 128,                          
        "with_likelihood": False
    },
    "run_type": "sampling",
    "version": 2
  }
  conf_filename = os.path.join(agent_dir, "evaluate_agent_config.json")
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename
