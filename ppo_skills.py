# PPO_Skills.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker  # <-- ensures masks used in rollout

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from top_down_env_gymnasium import CraftaxTopDownEnv
from top_down_env_gymnasium_options import OptionsOnTopEnv

import imageio


import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from sb3_contrib.common.wrappers import ActionMasker

from option_helpers import FixedSeedAlways, to_gif_frame
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_primitives", type=int, default=16)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_skill_len", type=int, default=25)

parser.add_argument("--skill_list", nargs="+", default=['wood_pick'])
parser.add_argument("--root", type=str, default='Traces/stone_pickaxe_easy')
parser.add_argument("--bc_checkpoint_dir", type=str, default='bc_checkpoints_resnet')
parser.add_argument("--dataset_mean_std_path", type=str, default='dataset_mean_std.npy')

parser.add_argument("--run_name", type=str, default='test_ppo_options')
parser.add_argument("--ppo_seed", type=int, default=888)
parser.add_argument("--target_primitive_steps", type=int, default=150_000)
parser.add_argument("--max_decision_steps", type=int, default=1_000_000)

args, _ = parser.parse_known_args()

class PrimitiveStepStopper(BaseCallback):
    """
    Stops training once the tracked primitive environment steps reach the target.
    Relies on OptionsOnTopEnv injecting `primitive_steps` into the info dict.
    Tracks cumulative steps across all episodes (since primitive_steps resets per episode).
    """

    def __init__(self, target_primitive_steps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.target = int(target_primitive_steps)
        self._cumulative_steps = 0
        self._last_primitive_steps = {}  # per-env tracking

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", []) if "dones" in self.locals else [False] * len(infos)
        
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            primitive_steps = info.get("primitive_steps")
            if primitive_steps is None:
                continue
            
            primitive_steps = int(primitive_steps)
            last_val = self._last_primitive_steps.get(env_idx, 0)
            done = dones[env_idx] if env_idx < len(dones) else False
            
            # If primitive_steps decreased or episode ended, we had a reset
            # Add the last episode's total to cumulative
            if primitive_steps < last_val or done:
                if last_val > 0:  # Only add if we had steps in the previous episode
                    self._cumulative_steps += last_val
                    if self.verbose > 0 and self.n_calls % 1000 == 0:
                        print(
                            f"[PrimitiveStepStopper] Episode ended: added {last_val} steps. "
                            f"Total cumulative: {self._cumulative_steps}"
                        )
            
            # Update tracking
            self._last_primitive_steps[env_idx] = primitive_steps
            
            # Current cumulative = sum of completed episodes + current episode steps
            current_total = self._cumulative_steps + primitive_steps
            
            # Periodic logging
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                print(
                    f"[PrimitiveStepStopper] Step {self.n_calls}: "
                    f"Current episode: {primitive_steps}, "
                    f"Cumulative: {self._cumulative_steps}, "
                    f"Total: {current_total}/{self.target}"
                )
            
            if current_total >= self.target:
                if self.verbose > 0:
                    print(
                        f"[PrimitiveStepStopper] Target reached: "
                        f"{current_total} primitive steps (target {self.target})."
                    )
                return False
        return True

    def get_cumulative_primitive_steps(self) -> int:
        """Get the total cumulative primitive steps across all episodes."""
        current_steps = sum(self._last_primitive_steps.values())
        return self._cumulative_steps + current_steps


def make_options_env(*, seed: int, render_mode=None,  max_episode_steps=100):
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
            num_primitives=args.num_primitives,
            gamma=args.gamma,
            max_skill_len=args.max_skill_len,
            skill_list=args.skill_list,
            root=args.root,
            bc_checkpoint_dir=args.bc_checkpoint_dir,
            dataset_mean_std_path=args.dataset_mean_std_path,
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
    K = 5
    TRAIN_SEED = 888  # set your fixed training seed here

    train_env = DummyVecEnv([make_options_env(seed=TRAIN_SEED, render_mode=None)])
    train_env = VecTransposeImage(train_env) 
    train_env = VecMonitor(train_env)

    print("Training PPO on options to get wood_pickaxe SEED : ", args.ppo_seed)
    print(f"Target primitive steps: {args.target_primitive_steps}")

    model = MaskablePPO(
        "CnnPolicy",                   # pixels -> CNN
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",
        seed=args.ppo_seed,
    )



    model.learn(
        total_timesteps=args.max_decision_steps,
        tb_log_name=args.run_name,   
        log_interval=1,
        progress_bar=True,
        callback=PrimitiveStepStopper(
            target_primitive_steps=args.target_primitive_steps,
            verbose=1,
        ),
    )
    model.save(args.run_name)

