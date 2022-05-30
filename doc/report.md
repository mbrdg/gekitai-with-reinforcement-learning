---
title:
- 'Gekitai with Reinforcement Learning'
author:
- João Sousa
- Miguel Rodrigues
- Ricardo Ferreira
date: 
- May 31, 2022
---

## Specification

- In this assignment the main goal is to develop an AI capable of playing
gekitai using reinfocement learning algorithms.

- Since the gekitai game is very simple, the goal for our agent is to win
games against a more traditional algorithms (in particular MCTS).

## Tools and algorithms

- For this project, we choose python as the main programming language, since it
offers a lot of utilitaries and lots of libraries targeted to RL projects.

- For the environment we used OpenAI [gym](https://www.gymlibrary.ml/).

  - This was a challenge since gym API is best suited for single-agent
  environments. This means that `step()` would require some adaptation, i.e.
  `step()` would play for both the agent and its opponent.

- The implementation of the RL algorithms will be provided by
[Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/).

## Implementation Scheduling

- At the moment, we have already implemented the environment for the gekitai
game.

- The next step is to make our agent learn using the appropriate RL algorithms,
such as:

  - DQN
  - PPO
  - etc.

## References

- Some of the references for the work already carried out:

  - [Gekitai Rules](https://boardgamegeek.com/boardgame/295449/gekitai)
  - [IA's course page @ moodle](https://moodle.up.pt/course/view.php?id=4088)
  - [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
  - [OpenAI gym](https://www.gymlibrary.ml/)
  - [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)

