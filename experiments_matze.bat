@ECHO OFF
setlocal
set PYTHONPATH=C:\\Users\\Matthias\\deep-rl-drones
python ./experiments/learning/singleagent.py --env loop --act rpm --algo a2c --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env loop --act rpm --algo a2c --use_advanced_loss --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env flips --act rpm --algo a2c --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env flips --act rpm --algo a2c --use_advanced_loss --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env loop --act rpm --algo ppo --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env flips --act rpm --algo ppo --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env loop --act rpm --algo ppo --use_advanced_loss --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env flips --act rpm --algo ppo --use_advanced_loss --timesteps 500000 --save_subdir final
endlocal
