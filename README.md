# Gekitai with Reiforcement Learning

The [Gekitai](https://boardgamegeek.com/boardgame/295449/gekitai) game with
OpenAI gym for reinforcement learning

## Dependencies

The required dependecies are specified in the `Pipfile` file, meaning
it is only required to have [pipenv](https://pipenv.pypa.io/en/latest/)
installed in the system.

After the installation or if you already have 
[pipenv](https://pipenv.pypa.io/en/latest/) run the commands below to install
all the required dependencies:

```bash
$ pipenv --python=3.10
$ pip install 'stable-baselines3[extra]' scipy jupyter
```

## Candy Stuff :lollipop:

Inside this [notebook](gekitai-with0rl.ipynb) there is already code that will
generate models with the help of the RL algorithms. During that processing the
Stable Baselines3 will also log to [TensorBoard](https://www.tensorflow.org/tensorboard)
some graphs that you may find useful.

For launching an instance of TensorBoard run the following commands:

```bash
$ pipenv shell
$ tensorboard --logdir=logs
```

After that a link to localhost should appear, et voilà!

## Contributors

- [João Sousa](mailto:up201904739@edu.fc.up.pt)
- [Miguel Rodrigues](mailto:up201906042@edu.fe.up.pt)
- [Ricardo Ferreira](mailto:up201907835@edu.fe.up.pt)

