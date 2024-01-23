import multi_ppo as mp
import multi_ppo_coop as mpc

from easydict import EasyDict

myconfig = {
        'n_env': 64,
        'num_snakes': 2,
        'width': 20,
        'height': 20,
        'vision_range': 5,
        'frame_stack': 2,
        'batch_size': 512,
        'epoch': 4,
		'max_steps': int(1e6),
        'train_interval': 512,
        'log_level': 10,
        'log_interval': 5000,
        'save_interval': 100000,
        'lr': 1e-4,
        'gamma': 0.99,
        'grad_clip': 10,
        'tag': 'DATA',
        'custom_rewardf': {
            'fruit': 1.0,
            'kill': -1.0,
            'lose': -10.0,
            'win': 0.0,
            'time': -0.05
        }
    }

if __name__ == "__main__":
	config = EasyDict(myconfig)

	config.num_snakes = 2
	config.custom_rewardf.time = -0.05
	mp.train(config)
	config.custom_rewardf.time = -0.01
	mp.train(config)
	# config.custom_rewardf.time = 0.01
	# mp.train(config)
	# config.custom_rewardf.time = 0.05
	# mp.train(config)

	config.num_snakes = 3
	config.custom_rewardf.time = -0.05
	mp.train(config)
	config.custom_rewardf.time = -0.01
	mp.train(config)
	# config.custom_rewardf.time = 0.01
	# mp.train(config)
	# config.custom_rewardf.time = 0.05
	# mp.train(config)

	config.num_snakes = 4
	config.custom_rewardf.time = -0.05
	mp.train(config)
	config.custom_rewardf.time = -0.01
	mp.train(config)
	# config.custom_rewardf.time = 0.01
	# mp.train(config)
	# config.custom_rewardf.time = 0.05
	# mp.train(config)

	##########################################################

	config.num_snakes = 2
	# config.custom_rewardf.time = -0.05
	# mpc.train(config)
	# config.custom_rewardf.time = -0.01
	# mpc.train(config)
	config.custom_rewardf.time = 0.01
	mpc.train(config)
	# config.custom_rewardf.time = 0.05
	# mpc.train(config)

	config.num_snakes = 3
	# config.custom_rewardf.time = -0.05
	# mpc.train(config)
	# config.custom_rewardf.time = -0.01
	# mpc.train(config)
	config.custom_rewardf.time = 0.01
	mpc.train(config)
	# config.custom_rewardf.time = 0.05
	# mpc.train(config)

	config.num_snakes = 4
	# config.custom_rewardf.time = -0.05
	# mpc.train(config)
	# config.custom_rewardf.time = -0.01
	# mpc.train(config)
	config.custom_rewardf.time = 0.01
	mpc.train(config)
	# config.custom_rewardf.time = 0.05
	# mpc.train(config)
