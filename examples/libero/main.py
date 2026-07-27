import collections
import dataclasses
import logging
import math
import pathlib
import csv
from typing import Literal, Optional

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


import matplotlib.pyplot as plt
import os

try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as exc:
    if exc.name != "libero":
        raise
    raise ModuleNotFoundError(
        "LIBERO is not installed in this Python environment. Follow examples/libero/README.md: "
        "create examples/libero/.venv, install examples/libero/requirements.txt and "
        "third_party/libero/requirements.txt, install -e third_party/libero, then run with "
        "PYTHONPATH=$PWD/third_party/libero."
    ) from exc


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    server_wait_timeout_s: float = 120.0
    resize_size: int = 224
    replan_steps: int = 5
    server_input_mode: Literal["libero", "go2"] = "libero"
    action_dim: int = 7

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    exp_name: str = ("debug")
    resume_id: int = 0
    task_start: int = 0
    task_count: Optional[int] = None
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    max_steps: Optional[int] = None
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "./libero_videos"  # Path to save videos
    action_log_path: Optional[str] = None  # Optional CSV path for debugging predicted action chunks
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220 * 3
    elif args.task_suite_name == "libero_object":
        max_steps = 280 * 3
    elif args.task_suite_name == "libero_goal":
        max_steps = 300 * 3
    elif args.task_suite_name == "libero_10":
        max_steps = 520 * 3
    elif args.task_suite_name == "libero_90":
        max_steps = 400 * 3
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps is not None:
        max_steps = args.max_steps
    video_out_path = args.video_out_path
    video_out_path_per_task = pathlib.Path(video_out_path) / args.exp_name / args.task_suite_name
    video_out_path_per_task_success = video_out_path_per_task / "success"
    video_out_path_per_task_failure = video_out_path_per_task / "failure"

    pathlib.Path(video_out_path_per_task_success).mkdir(parents=True, exist_ok=True)
    pathlib.Path(video_out_path_per_task_failure).mkdir(parents=True, exist_ok=True)
    action_log_file = None
    action_log_writer = None
    if args.action_log_path is not None:
        action_log_path = pathlib.Path(args.action_log_path)
        action_log_path.parent.mkdir(parents=True, exist_ok=True)
        action_log_file = action_log_path.open("w", newline="")
        action_log_writer = csv.DictWriter(
            action_log_file,
            fieldnames=[
                "task_id",
                "episode_idx",
                "t",
                "eef_x",
                "eef_y",
                "eef_z",
                "state_gripper",
                "chunk_len",
                "exec_gripper_min",
                "exec_gripper_max",
                "exec_gripper_mean",
                "exec_gripper_first",
                "exec_gripper_values",
                "first_action",
            ],
        )
        action_log_writer.writeheader()

    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        wait_timeout_s=args.server_wait_timeout_s,
        retry_interval_s=1.0,
    )
    logging.info(f"Policy server metadata: {client.get_server_metadata()}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_stop = num_tasks_in_suite if args.task_count is None else min(num_tasks_in_suite, args.task_start + args.task_count)
    for task_id in tqdm.tqdm(range(args.task_start, task_stop)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # resume logic
            if total_episodes < args.resume_id:
                task_episodes += 1
                total_episodes += 1
                continue


            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        element = _make_policy_observation(
                            img,
                            wrist_img,
                            state,
                            str(task_description),
                            input_mode=args.server_input_mode,
                        )

                        # Query model to get action
                        ret_result = client.infer(element)
                        action_chunk = np.asarray(ret_result["actions"])[..., : args.action_dim]

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        if action_log_writer is not None:
                            exec_chunk = action_chunk[: args.replan_steps]
                            gripper = exec_chunk[:, -1]
                            action_log_writer.writerow(
                                {
                                    "task_id": task_id,
                                    "episode_idx": episode_idx,
                                    "t": t,
                                    "eef_x": state[0],
                                    "eef_y": state[1],
                                    "eef_z": state[2],
                                    "state_gripper": state[-1],
                                    "chunk_len": len(action_chunk),
                                    "exec_gripper_min": float(np.min(gripper)),
                                    "exec_gripper_max": float(np.max(gripper)),
                                    "exec_gripper_mean": float(np.mean(gripper)),
                                    "exec_gripper_first": float(gripper[0]),
                                    "exec_gripper_values": " ".join(f"{x:.6g}" for x in gripper),
                                    "first_action": " ".join(f"{x:.6g}" for x in exec_chunk[0]),
                                }
                            )
                            action_log_file.flush()
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if suffix == "failure":
                imageio.mimwrite(
                    video_out_path_per_task_failure / f"rollout_{task_id}_{episode_idx}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            if suffix == "success":
                imageio.mimwrite(
                    video_out_path_per_task_success / f"rollout_{task_id}_{episode_idx}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if action_log_file is not None:
        action_log_file.close()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _make_policy_observation(
    image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    prompt: str,
    *,
    input_mode: str,
) -> dict:
    if input_mode == "libero":
        return {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": prompt,
        }
    if input_mode == "go2":
        return {
            "images": {
                "top_head": image,
                "hand_left": wrist_image,
                "hand_right": np.zeros_like(image),
            },
            "state": state,
            "prompt": prompt,
        }
    raise ValueError(f"Unsupported server_input_mode: {input_mode}")


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
