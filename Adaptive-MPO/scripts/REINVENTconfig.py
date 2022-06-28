import json
import numpy as np

def parse_config_file(f, scoring_component_names):
    configuration = json.load(open(f))
    configuration_scoring_components = {}
    for comp in configuration["parameters"]["scoring_function"]["parameters"]:
        configuration_scoring_components[comp["name"]] = comp
    low = np.array([configuration_scoring_components[comp]["specific_parameters"]["transformation"]["low"] for comp in scoring_component_names])
    high = np.array([configuration_scoring_components[comp]["specific_parameters"]["transformation"]["high"] for comp in scoring_component_names])
    # double sigmoid params:
    coef_div = np.ones(high.shape)
    coef_si = np.ones(high.shape)
    coef_se = np.ones(high.shape)
    for i, comp in enumerate(scoring_component_names):
        try:
            coef_div[i] = configuration_scoring_components[comp]["specific_parameters"]["transformation"]["coef_div"] 
            coef_si[i] = configuration_scoring_components[comp]["specific_parameters"]["transformation"]["coef_si"]
            coef_se[i] = configuration_scoring_components[comp]["specific_parameters"]["transformation"]["coef_se"]
        except KeyError:
            # In case original transformations were not double sigmoid, use default values
            coef_div[i] = high[i]
            coef_si[i] = 10
            coef_se[i] = 10
    return {'low': low, 'high': high, 'coef_div': coef_div, 'coef_si': coef_si, 'coef_se': coef_se}
