from src.utils.environment_design import EnvironmentDesign

def run_all_methods(base_environment,
                    user_params,
                    learn_what,
                    parameter_ranges_dict,
                    methods: list,
                    num_runs_per_method,
                    candidate_environment_args_methods: dict
                    )
    
    for method in methods:

        