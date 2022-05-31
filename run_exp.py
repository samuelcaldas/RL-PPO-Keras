from env import Env
from ppo import Agent


def main(dictionary_agent_configuration, dictionary_env_configuration, dictionary_exp_configuration, dictionary_path):
    env = Env(dictionary_env_configuration)

    dictionary_agent_configuration["ACTION_DIM"] = env.action_dim
    dictionary_agent_configuration["STATE_DIM"] = (env.state_dim, )

    agent = Agent(dictionary_agent_configuration, dictionary_path, dictionary_env_configuration)

    for count_episode in range(dictionary_exp_configuration["TRAIN_ITERATIONS"]):
        state = env.reset()
        reward_sum = 0
        for count_step in range(dictionary_exp_configuration["MAX_EPISODE_LENGTH"]):
            if count_episode > dictionary_exp_configuration["TRAIN_ITERATIONS"] - 10:
                env.render()

            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)

            reward /= 100
            reward_sum += reward
            if done:
                reward = -1

            agent.store_transition(state, action, new_state, reward, done)
            if count_step % dictionary_agent_configuration["BATCH_SIZE"] == 0 and count_step != 0:
                agent.train_network()
            state = new_state

            if done:
                break

            if count_step % 10 == 0:
                print("Episode:{}, step:{}, r_sum:{}".format(count_episode, count_step, reward_sum))


