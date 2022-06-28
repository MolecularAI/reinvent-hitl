from scripts.write import write_sample_file
import os

def sample_mols_from_agent(jobid, jobname, agent_dir, reinvent_env, reinvent_dir, N=1000):
    print("Sampling from agent " + os.path.join(agent_dir, "Agent.ckpt"))
    conf_file = write_sample_file(jobid, jobname, agent_dir, N)
    os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + conf_file + '&> ' + agent_dir + '/sampling.err')


