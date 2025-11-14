#PPO_Basic.py

import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

from option_helpers import FixedSeedAlways, to_gif_frame



parser = argparse.ArgumentParser()
parser.add_argument("--ppo_seed", type=int, default=888)
parser.add_argument("--run_name", type=str, default="ppo_craftax_wood_pick_actions")
parser.add_argument("--target_primitive_steps", type=int, default=300_000)
parser.add_argument("--max_decision_steps", type=int, default=500_000)

args, _ = parser.parse_known_args()


class PrimitiveStepWrapper(gym.Wrapper):
    """
    Wrapper for primitive-only environments that adds primitive_steps to info dict.
    For primitive-only, each step is 1 primitive step.
    """
    def __init__(self, env):
        super().__init__(env)
        self.total_primitive_steps = 0
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.total_primitive_steps = 0
        info = dict(info) if info is not None else {}
        info["primitive_steps"] = self.total_primitive_steps
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_primitive_steps += 1
        info = dict(info) if info is not None else {}
        info["primitive_steps"] = self.total_primitive_steps
        return obs, reward, terminated, truncated, info


class PrimitiveStepStopper(BaseCallback):
    """
    Stops training once the tracked primitive environment steps reach the target.
    Relies on PrimitiveStepWrapper injecting `primitive_steps` into the info dict.
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


class PrimitiveStepLogger(BaseCallback):
    """
    Logs TensorBoard metrics using primitive steps as the x-axis instead of decision steps.
    Maintains a sliding window of episode statistics (like SB3's Monitor) and logs their averages.
    """
    
    def __init__(self, verbose: int = 0, log_interval: int = 100, window_size: int = 100):
        super().__init__(verbose=verbose)
        self.log_interval = log_interval
        self.window_size = window_size
        self._cumulative_steps = 0
        self._last_primitive_steps = {}
        self._last_log_step = 0
        
        # Buffers for sliding window averages (like SB3's Monitor)
        self.episode_returns = []
        self.episode_lengths_decision = []
        self.episode_lengths_primitive = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", []) if "dones" in self.locals else [False] * len(infos)
        
        # Track cumulative primitive steps and collect episode stats on episode end
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            primitive_steps = info.get("primitive_steps")
            if primitive_steps is None:
                continue
            
            primitive_steps = int(primitive_steps)
            last_val = self._last_primitive_steps.get(env_idx, 0)
            done = dones[env_idx] if env_idx < len(dones) else False
            
            # On episode end, collect stats and add to buffers
            if done and "episode" in info:
                ep_info = info["episode"]
                ep_return = ep_info.get("r")
                ep_len_decision = ep_info.get("l")
                ep_len_primitive = primitive_steps  # This is the primitive steps for the completed episode
                
                if ep_return is not None:
                    self.episode_returns.append(float(ep_return))
                    if len(self.episode_returns) > self.window_size:
                        self.episode_returns.pop(0)
                
                if ep_len_decision is not None:
                    self.episode_lengths_decision.append(float(ep_len_decision))
                    if len(self.episode_lengths_decision) > self.window_size:
                        self.episode_lengths_decision.pop(0)
                
                if ep_len_primitive is not None:
                    self.episode_lengths_primitive.append(int(ep_len_primitive))
                    if len(self.episode_lengths_primitive) > self.window_size:
                        self.episode_lengths_primitive.pop(0)
            
            # Update cumulative tracking
            if primitive_steps < last_val or done:
                if last_val > 0:
                    self._cumulative_steps += last_val
            
            self._last_primitive_steps[env_idx] = primitive_steps
        
        current_total = self._cumulative_steps + sum(self._last_primitive_steps.values())
        
        # Log metrics periodically based on primitive steps
        if current_total - self._last_log_step >= self.log_interval:
            try:
                # Compute means over sliding window (like SB3's Monitor)
                ep_rew_mean = np.mean(self.episode_returns) if len(self.episode_returns) > 0 else None
                ep_len_decision_mean = np.mean(self.episode_lengths_decision) if len(self.episode_lengths_decision) > 0 else None
                ep_len_primitive_mean = np.mean(self.episode_lengths_primitive) if len(self.episode_lengths_primitive) > 0 else None
                
                # Log to TensorBoard with primitive steps as x-axis
                if ep_rew_mean is not None:
                    self.logger.record("rollout_primitive/ep_rew_mean", ep_rew_mean)
                if ep_len_decision_mean is not None:
                    self.logger.record("rollout_primitive/ep_len_decision_mean", ep_len_decision_mean)
                if ep_len_primitive_mean is not None:
                    self.logger.record("rollout_primitive/ep_len_primitive_mean", ep_len_primitive_mean)
                
                # Also log the primitive step count itself
                self.logger.record("rollout_primitive/total_primitive_steps", current_total)
                
                # Dump to TensorBoard with primitive steps as the step counter
                self.logger.dump(step=current_total)
                self._last_log_step = current_total
            except Exception as e:
                if self.verbose > 0:
                    print(f"[PrimitiveStepLogger] Warning: Could not log metrics: {e}")
        
        return True


def make_env(seed: int):
    def _thunk():
        env = CraftaxTopDownEnv(
            seed=seed,               # initialize env RNG deterministically
            render_mode=None,
            reward_items=[],
            done_item="wood_pickaxe",
            include_base_reward=False,
            return_uint8=True,
        )
        env = TimeLimit(env, max_episode_steps=100)
        env = FixedSeedAlways(env, seed=seed)  # keep the SAME seed every reset
        env = PrimitiveStepWrapper(env)  # Add primitive_steps tracking
        env = RecordEpisodeStatistics(env)  # Add episode statistics
        return env
    return _thunk

if __name__ == "__main__":
    SEED = 888  # single training seed; every episode uses this seed

    train_env = DummyVecEnv([make_env(seed=SEED)])
    train_env = VecTransposeImage(train_env)  # HWC -> CHW for CnnPolicy
    train_env = VecMonitor(train_env)

    # model = PPO.load("ppo_craftax_wood_ppo_actions")

    print("Training PPO on basic actions to get wood_pickaxe SEED : ", args.ppo_seed)
    print(f"Target primitive steps: {args.target_primitive_steps}")

    # Optimize PPO for limited decision steps (~1,700 with 250k primitive steps)
    # With ~146 primitive steps per decision step, we need efficient learning
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        n_steps=128,                   # Reduced from default 2048: allows ~13 rollouts instead of 1
        # n_epochs=30,                   # Increased from default 10: more learning per rollout
        batch_size=64,                 # Smaller batches for more frequent updates
        learning_rate=3e-4,            # Slightly higher LR for faster learning
        gamma=1.0,                     # Use gamma=1.0 for fairness (no discounting asymmetry)
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",
        seed=args.ppo_seed,  # set SB3/torch RNGs too for full determinism
    )

    # Combine callbacks: stopper + logger
    stopper = PrimitiveStepStopper(
        target_primitive_steps=args.target_primitive_steps,
        verbose=1,
    )
    logger = PrimitiveStepLogger(verbose=1, log_interval=100)
    callback = CallbackList([stopper, logger])

    model.learn(
        total_timesteps=args.max_decision_steps,
        log_interval=1,
        tb_log_name=f"{args.run_name}",
        progress_bar=True,
        callback=callback,
    )
    
    model.save("ppo_craftax_wood_pick_actions")

    # -------- Eval vec env (choose a seed; use same wrapper for fixed-seed eval) --------
    

    # obs = eval_env.reset()
    # images = [to_gif_frame(obs)]

    # done = False
    # steps = 0
    # while not done and steps < 100:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, infos = eval_env.step(action)
    #     images.append(to_gif_frame(obs))
    #     done = bool(dones[0])

    #     print("Step:", steps, "Action:", action, "Reward:", rewards, "Done:", done)

    #     steps += 1

    # imageio.mimsave(f"craftax_ppo_wood_pick_actions_eval.gif", images, fps=1)
