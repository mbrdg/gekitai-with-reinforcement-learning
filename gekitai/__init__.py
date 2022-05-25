from gym.envs.registration import register

register(id='gekitai-v0',
         entry_point='gekitai.envs:GekitaiEnv')
