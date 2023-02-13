@ECHO OFF
setlocal
set PYTHONPATH=D:\\Uni Master\\Semester 3 (Sapienza)\\Reinforcement Learning\\project\\gym-pybullet-drones-0.5.2
python ./experiments/learning/singleagent.py --timesteps 150000 --algo ppo --save_subdir final
python ./experiments/learning/singleagent.py --timesteps 150000 --algo ourppo --save_subdir final
python ./experiments/learning/singleagent.py --timesteps 150000 --algo a2c --save_subdir final
python ./experiments/learning/singleagent.py --timesteps 150000 --algo ddpg --save_subdir final
python ./experiments/learning/singleagent.py --act rpm --algo ppo --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --act rpm --algo a2c --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --act rpm --algo ppo --use_advanced_loss --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --act rpm --algo a2c --use_advanced_loss --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env forward --act rpm --algo ppo --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env forward --act rpm --algo a2c --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env forward --act rpm --algo ppo --use_advanced_loss --timesteps 500000 --save_subdir final
python ./experiments/learning/singleagent.py --env forward --act rpm --algo a2c --use_advanced_loss --timesteps 500000 --save_subdir final
endlocal