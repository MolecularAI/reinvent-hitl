import reinvent_scoring

import numpy as np

def query(smiles):
  expert_model_path = './data/drd2.pkl'
  qsar_model ={ "component_type": "predictive_property",
                "name": "DRD2",
                "weight": 1,
                "model_path": expert_model_path,
                "smiles": [],
                "specific_parameters": {
                    "transformation_type": "no_transformation",
                    "scikit": "classification",
                    "transformation": False,
                    "descriptor_type": "ecfp",
                    "size": 2048,
                    "radius": 3
                }
              }
  scoring_function = {
    "name": "custom_sum",
    "parallel": True,
    "parameters": [
      qsar_model
    ]
  }
  scoring_function_parameters = reinvent_scoring.scoring.ScoringFuncionParameters(**scoring_function)
  expert_scoring_function = reinvent_scoring.scoring.ScoringFunctionFactory(scoring_function_parameters)
  result=expert_scoring_function.get_final_score(smiles)
  return result.total_score