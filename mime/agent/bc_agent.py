import os
import click
import gym
import json

import pickle as pkl

from bc.agent import BCAgent
from bc.utils.videos import write_video


@click.command(help="net_agent env_name [options]")
@click.argument("env_name", type=str)
@click.option("--seed", "-s", default=0, help="seed")
@click.option("-np", "--net_path", type=str, required=True, help="net path")
@click.option(
    "-ne", "--net_epoch", type=str, default="current", help="net epoch to load"
)
@click.option(
    "-v", "--video", type=str, default="", help="record a video of the demonstration"
)
@click.option(
    "-va",
    "--video-agent",
    type=str,
    default="",
    help="record a video of the demonstration",
)
@click.option(
    "-pa",
    "--pickle-actions/--no-pickle-actions",
    default=False,
    help="whether to pickle the actions",
)
@click.option(
    "-pn", "--pickle-name", type=str, default=None, help="name of the pickled actions"
)
@click.option("-d", "--device", type=str, default="cuda", help="cuda or cpu")
@click.option(
    "-tsc",
    "--timescales",
    type=json.loads,
    default='{"0": 50, "1": 50, "2":200, "3":200}',
    help="timescale(s) of the skill choice",
)
@click.option(
    "-cv2/-ncv2",
    "--use-cv2/--no-use-cv2",
    default=False,
    help="whether to show images with opencv",
)
@click.option(
    "-r/-nr", "--render/--no-render", default=True, help="whether to render the agent"
)
@click.option(
    "-ss",
    "--skill-sequence",
    type=json.loads,
    default="[0, 2, 3, 1, 2, 3]",
    # @click.option('-ss', '--skill-sequence', type=json.loads, default="[6, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]",
    help="skill sequence (if a hierarchical bc policy is used)",
)
@click.option(
    "-ia",
    "--image-augmentation",
    type=str,
    default="",
    help="image augmentation to use with the agent",
)
def main(
    env_name,
    seed,
    net_path,
    net_epoch,
    device,
    video,
    pickle_actions,
    video_agent,
    pickle_name,
    timescales,
    use_cv2,
    render,
    skill_sequence,
    image_augmentation,
):
    if use_cv2:
        import cv2

    env = gym.make(env_name)
    scene = env.unwrapped.scene
    if render:
        scene.renders(True)
    env.seed(seed)
    obs = env.reset()
    done = False
    action, skill = 0, None
    frames, frames_agent, actions = [], [], []
    step, skill_counter, step_after_skill = 0, 0, 0

    agent = BCAgent(
        net_path,
        net_epoch,
        max_steps=-1,
        device=device,
        augmentation_str=image_augmentation,
    )
    is_skill_policy = (
        "skills" in agent.model.args["mode"] or "film" in agent.model.args["mode"]
    )

    if is_skill_policy and isinstance(timescales, int):
        timescale_int = timescales
        timescales = {}
        for skill in range(max(skill_sequence) + 1):
            timescales[skill] = timescale_int
    elif isinstance(timescales, dict):
        keys = list(timescales.keys()).copy()
        for key in keys:
            timescales[int(key)] = timescales.pop(key)

    while action is not None:
        print(action)
        if step % 10 == 0:
            print("Step {}".format(step))
        if is_skill_policy and (
            step_after_skill == 0
            or (skill is not None and step_after_skill == timescales[skill])
        ):
            if skill_counter == len(skill_sequence):
                break
            skill = skill_sequence[skill_counter]
            print("Switching to the skill {} at step {}".format(skill, step))
            skill_counter += 1
            step_after_skill = 0

        action = agent.get_action(obs, skill)
        actions.append(action)
        obs, reward, done, info = env.step(action)

        step += 1
        step_after_skill += 1
        # rate.sleep()
        if use_cv2:
            cv2.imshow("rgb", obs["rgb0"][..., ::-1])
            cv2.waitKey(1)

        if video:
            im_cv2 = obs["rgb0"]
            im_agent = agent._stack_frames[-1]
            frames.append(im_cv2)
            frames_agent.append(((im_agent.numpy() + 1) / 2 * 255)[..., None])
            if step % 50 == 0:
                write_video(frames, video)
                if video_agent:
                    write_video(frames_agent, video_agent)

    if video:
        write_video(frames, video)
        if video_agent:
            write_video(frames_agent, video_agent)
    if pickle_actions:
        rollouts_dir = os.path.join(net_path, "rollouts")
        if not os.path.isdir(rollouts_dir):
            os.mkdir(rollouts_dir)
        if pickle_name is None:
            rollout_path = os.path.join(rollouts_dir, "{}.pkl".format(seed))
        else:
            rollout_path = os.path.join(
                rollouts_dir, "{}_{}.pkl".format(pickle_name, seed)
            )
        pkl.dump(actions, open(rollout_path, "wb"))
        print("actions are pickled to {}".format(rollout_path))

    if action is None:
        info["failure_message"] = "End of Script."
    if not info["success"]:
        click.secho(
            "Failure Seed {}: {}".format(seed, info["failure_message"]), fg="red"
        )

    print("Success", info["success"])


if __name__ == "__main__":
    main()
