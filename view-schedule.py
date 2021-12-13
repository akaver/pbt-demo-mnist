from ray.tune.schedulers import PopulationBasedTrainingReplay

policy = 'b9e1a_00013'
replay = PopulationBasedTrainingReplay(
    f"./data/FashionMNIST-pbt/pbt_policy_{policy}.txt")

print(replay.config)  # Initial config
  # Schedule, in the form of tuples (step, config)

print(replay._policy)


{'progress_bar_refresh_rate': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.001, 'batch_size': 32, 'data_dir': '~/mldata', 'data_mean': 0.28604063391685486, 'data_std': 0.35302430391311646,
 'augmentations': [['blur', 0.8259305059834876], ['rotate_left', 0.06433722300157074], ['rotate_right', 0.7939276445192464]]}
[(3, {'progress_bar_refresh_rate': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.001, 'batch_size': 32, 'data_dir': '~/mldata', 'data_mean': 0.28604063391685486, 'data_std': 0.35302430391311646,
      'augmentations': [['blur', 0.3133976561550309], ['rotate_left', 0.5569725831138659], ['rotate_right', 0.3569103635573554]]}),
 (9, {'progress_bar_refresh_rate': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.001, 'batch_size': 32, 'data_dir': '~/mldata', 'data_mean': 0.28604063391685486, 'data_std': 0.35302430391311646,
      'augmentations': [['blur', 0.8471038256796716], ['rotate_left', 0.7615997305311374], ['rotate_right', 0.7064858680200432]]}),
 (15, {'progress_bar_refresh_rate': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.001, 'batch_size': 32, 'data_dir': '~/mldata', 'data_mean': 0.28604063391685486, 'data_std': 0.35302430391311646,
       'augmentations': [['blur', 0.16451135950923235], ['rotate_left', 0.7900890030913503], ['rotate_right', 0.8860171664725945]]})]
