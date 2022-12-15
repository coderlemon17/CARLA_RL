# Carla-RL
在Carla仿真平台进行强化学习算法
# docs
此文件夹中包含文档：Carla编译版部署和Carla0.9.6部署  
    Carla编译版部署：主要介绍了部署Carla编译版本步骤，另外还包含安装显卡驱动、安装conda环境、安装ros、安装ros-bridge等安装流程；  
    Carla0.9.6部署：为Carla Releases版本部署，包含安装显卡驱动、安装conda环境、配置深度强化学习环境  

# How to run:
```bash
# in CARLA_0.9.6
./CarlaUE4.sh -carla-port=2000
# in gym-carla_lanekeep
xvfb-run -n 99 --server-args="-screen 0 1280x760x24" -l python Carla_sac.py
```

# TODOS
- [] Fix all absolute path setting and refactoring the code.