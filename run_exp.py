from env import Env
from ppo import Agent


def main(dictionary_agent_configuration, dictionary_env_configuration, dictionary_exp_configuration, dictionary_path):
    env = Env(dictionary_env_configuration)

    dictionary_agent_configuration["ACTION_DIM"] = env.action_dim
    dictionary_agent_configuration["STATE_DIM"] = (env.state_dim, )

    agent = Agent(dictionary_agent_configuration, dictionary_path, dictionary_env_configuration)

    for cnt_episode in range(dictionary_exp_configuration["TRAIN_ITERATIONS"]):
        state = env.reset()
        r_sum = 0
        for cnt_step in range(dictionary_exp_configuration["MAX_EPISODE_LENGTH"]):
            if cnt_episode > dictionary_exp_configuration["TRAIN_ITERATIONS"] - 10:
                env.render()

            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)

            reward /= 100
            r_sum += reward
            if done:
                reward = -1

            agent.store_transition(state, action, new_state, reward, done)
            if cnt_step % dictionary_agent_configuration["BATCH_SIZE"] == 0 and cnt_step != 0:
                agent.train_network()
            state = new_state

            if done:
                break

            if cnt_step % 10 == 0:
                print("Episode:{}, step:{}, r_sum:{}".format(cnt_episode, cnt_step, r_sum))


