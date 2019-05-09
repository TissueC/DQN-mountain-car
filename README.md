# 直接运行 run_MountainCar.py

# 调参

```
RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=0.02,output_graph=False)
```

# 可视化观察小车训练过程
```
env.render()
```
该操作会大大减慢程序运行速度
