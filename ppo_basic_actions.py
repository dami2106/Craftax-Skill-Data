import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

def to_gif_frame(obs):
    import numpy as np
    arr = np.asarray(obs)

    # remove vec batch dim if present (N=1)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]                    # (C,H,W)

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))   # (H,W,C)

    # if single channel, replicate to RGB
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)      # (H,W,3)

    # scale/clip & cast to uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        arr = arr.astype(np.uint8)

    # if still grayscale 2D, OK for GIF; otherwise ensure HWC
    return arr

# ---- Wrapper: always reset with the same given seed ----
class FixedSeedOnReset(gym.Wrapper):
    def __init__(self, env, seed: int):
        super().__init__(env)
        self._seed = int(seed)

    def reset(self, *, seed=None, options=None, **kwargs):
        # Ignore incoming seed and force a fixed one every episode
        kwargs.pop("seed", None)
        return self.env.reset(seed=self._seed, options=options, **kwargs)


def make_env(seed: int):
    def _thunk():
        env = CraftaxTopDownEnv(
            seed=seed,               # initialize env RNG deterministically
            render_mode=None,
            reward_items=[],
            done_item="wood_pickaxe",
            include_base_reward=False,
        )
        env = TimeLimit(env, max_episode_steps=25)
        env = FixedSeedOnReset(env, seed=seed)  # keep the SAME seed every reset
        return env
    return _thunk

if __name__ == "__main__":
    SEED = 1000  # single training seed; every episode uses this seed

    train_env = DummyVecEnv([make_env(seed=SEED)])
    train_env = VecTransposeImage(train_env)  # HWC -> CHW for CnnPolicy
    train_env = VecMonitor(train_env)

    model = PPO.load("ppo_craftax_wood_ppo_actions")

    # model = PPO(
    #     policy="CnnPolicy",
    #     env=train_env,
    #     seed=SEED,  # set SB3/torch RNGs too for full determinism
    #     verbose=1,
    #     tensorboard_log="./tb_logs_ppo_craftax",
    #     device="auto",
    # )

    # model.learn(
    #     total_timesteps=500_000,
    #     log_interval=10,
    #     tb_log_name="ppo_wood_actions",
    #     progress_bar=True,
    # )
    # model.save("ppo_craftax_wood_ppo_actions")

    # -------- Eval vec env (choose a seed; use same wrapper for fixed-seed eval) --------
    eval_env = DummyVecEnv([make_env(seed=SEED)])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)

    obs = eval_env.reset()
    images = [to_gif_frame(obs)]

    done = False
    steps = 0
    while not done and steps < 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        images.append(to_gif_frame(obs))
        done = bool(dones[0])

        print("Step:", steps, "Action:", action, "Reward:", rewards, "Done:", done)

        steps += 1
    
    images.append(to_gif_frame(obs))

    imageio.mimsave(f"craftax_run_test_wood_easy_ppo_actions_{done}.gif", images, fps=1)
