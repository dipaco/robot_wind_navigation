from environments import FormationEnv, TurbulentFormationEnv
import hydra
import numpy as np

@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    # instantiate the gym environment

    env = TurbulentFormationEnv(cfg)
    obs = env.reset()


    done = False
    while not done:
        n_agents = cfg.formation_params.num_nodes
        action = np.zeros((n_agents, 2))
        obs, reward, done, _ = env.step(action)
        env.render()




if __name__ == "__main__":
    main()
