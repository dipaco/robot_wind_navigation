# Aireal Robot formations in the presence of wind turbulence

## Installation


After cloning the repository, we need to initialize the [pytorch_sac](https://github.com/dipaco/pytorch_sac.git) as a submodule.

```
git submodule init
git submodule update
```

We recommend using the repository within a clean virtual environment. We provide the ```make_env.sh``` script to create a new environment with all the necesary packages to run the project. You just have to run:

```
conda create -n <env_name> python=3.8 --no-default-packages -y
conda activate <env_name>
./make_env.sh
./run.sh
```




