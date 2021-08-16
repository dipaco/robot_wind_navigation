Xvfb :99 -screen 0 640x480x24 &
export DISPLAY=:99

HYDRA_FULL_ERROR=1

python train.py $@
