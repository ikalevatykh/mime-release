import gym
from bc.agent import ScriptAgent, BCAgent


def create_agent(agent_type, env):
    if agent_type == "script":
        agent = ScriptAgent(env)
    elif agent_type == "bc":
        agent = BCAgent(model_dir, epoch, max_steps, device, image_augmentation)
        agent.seed_exp(seed)

    # for RGB: should be train and tested with the same egl setup
    if agent_type == "bc" and "rgb" in agent.model.args["input_type"]:
        agent_env_name = agent.model.args["env_name"]
        if "EGL" in agent_env_name:
            assert "EGL" in env_name
        else:
            assert "EGL" not in env_name


def get_action(agent, obs):
    if isinstance(agent, BCAgent):
        action = agent.get_action(obs, 0)
    else:
        action = agent.get_action()
    return action


def run_episode(env, agent_type, seed):
    env.seed(seed)
    obs = env.reset()
    agent = create_agent(agent_type, env)

    action = get_action(agent, obs)
    while action:
        obs, reward, done, info = env.step(action)
        if done:
            print("Success.")
            if len(info["failure_message"]) > 0:
                break

        action = get_action(agent, obs)

    if not info["success"]:
        if action is None:
            print("Failure: reached max steps.")


env_name = ""
agent_type = "script"

env = gym.make(env_name)
env_scene = env.unwrapped.scene
env_scene.renders(True)
seed = 0

run_episode(env, agent_type, seed)
