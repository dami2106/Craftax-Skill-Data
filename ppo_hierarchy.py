# PPO_hierarchy.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker  # <-- ensures masks used in rollout

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from top_down_env_gymnasium_hierarchy import CraftaxTopDownEnv
from top_down_env_gymnasium_hierarchy import OptionsOnTopEnv

import imageio


import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from sb3_contrib.common.wrappers import ActionMasker

from option_helpers import FixedSeedAlways, to_gif_frame


def make_options_env(*, seed: int, render_mode=None, max_episode_steps=100):
    def _thunk():
        base = CraftaxTopDownEnv(
            render_mode=render_mode,
            reward_items=[],
            done_item="wood_pickaxe",
            include_base_reward=False,
            return_uint8=True,
        )
        core = OptionsOnTopEnv(
            base,
            num_primitives=16,
            gamma=0.99,
            max_skill_len=25,
        )

        # 1) ActionMasker wraps the env that has `action_masks`
        def mask_fn(e):  # no unwrapping needed
            return e.action_masks()
        masked = ActionMasker(core, mask_fn)

        # 2) Add outer wrappers afterwards
        capped   = TimeLimit(masked, max_episode_steps=max_episode_steps)
        fixed    = FixedSeedAlways(capped, seed=seed)
        logged   = RecordEpisodeStatistics(fixed)
        return logged
    return _thunk


def mask_fn(env):
    return env.action_masks()


# Helper: unwrap wrappers to reach the env that exposes `action_masks()`
def get_action_masks(env_or_vec):
    """Return the current action mask from a (possibly vectorized and wrapped) env.

    Works with DummyVecEnv and common Gym wrappers by walking down `.env` until
    an object with `action_masks()` is found.
    """
    # If a VecEnv is provided, grab the first sub-env
    env = env_or_vec.envs[0] if hasattr(env_or_vec, "envs") else env_or_vec

    # Walk through wrapper chain
    cur = env
    while True:
        if hasattr(cur, "action_masks") and callable(getattr(cur, "action_masks")):
            return cur.action_masks()
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break

    # Final check (in case the base env itself had it but loop ended)
    if hasattr(cur, "action_masks") and callable(getattr(cur, "action_masks")):
        return cur.action_masks()

    raise AttributeError("No action_masks() found in env wrapper stack.")


if __name__ == "__main__":
    TRAIN_SEED = 888  # set your fixed training seed here

    train_env = DummyVecEnv([make_options_env(seed=TRAIN_SEED, render_mode=None)])
    train_env = VecTransposeImage(train_env) 
    train_env = VecMonitor(train_env)

    model = MaskablePPO(
        "CnnPolicy",                   # pixels -> CNN
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",
    )


    # For evaluation you can reuse the same seed or choose a different fixed seed
    EVAL_SEED = TRAIN_SEED
    eval_env_vec = DummyVecEnv([make_options_env(seed=EVAL_SEED, render_mode=None)])
    eval_env_vec = VecTransposeImage(eval_env_vec)
    eval_env_vec = VecMonitor(eval_env_vec)

    eval_cb = MaskableEvalCallback(
        eval_env_vec,
        eval_freq=500,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(
        total_timesteps=800,
        tb_log_name="ppo_wood_pick_hierarchy",   # TB subdir
        log_interval=10,
        progress_bar=True,
        callback=eval_cb,
    )
    model.save("ppo_craftax_wood_pick_hierarchy")
    obs = eval_env_vec.reset()
    frames = [to_gif_frame(obs)]

    done = False
    steps = 0
    while not done and steps < 100:
        # Pull masks from the FIRST sub-env (vectorized)
        masks = get_action_masks(eval_env_vec)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, info = eval_env_vec.step(action)
        frames.append(to_gif_frame(obs))
        done = bool(terminated[0])
        print("Step:", steps, "Action:", action, "Reward:", reward, "Done:", done)
        steps += 1

    imageio.mimsave("craftax_ppo_wood_pick_hierarchy_eval.gif", frames, fps=5)

