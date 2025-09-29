# options_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error

from top_down_env_gymnasium import CraftaxTopDownEnv

from option_helpers import *

class OptionsOnTopEnv(gym.Env):
    """
    Wraps CraftaxTopDownEnv to expose a Discrete action space:
      [0..P-1]      -> primitive actions (single step)
      [P..P+K-1]    -> options (macro-step via BC until termination)

    MaskablePPO will call `action_masks()` to know which actions are valid.
    Primitives are *always valid*; options are valid iff available_skills(obs)[i] is True.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(
        self,
        base_env,                     # an instance of CraftaxTopDownEnv
        num_primitives: int = 17,    
        gamma: float = 0.99,
        max_skill_len: int = 50,      # safety cap for option rollout
    ):
        super().__init__()
        self.env = base_env
        self.gamma = float(gamma)
        self.max_skill_len = int(max_skill_len)

        # Models / skills
        self.models = load_all_models_hierarchy(
        skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
        hierarchies_dir = 'Traces/stone_pickaxe_easy/hierarchy_data/Simple',
        symbol_map = {
            "0": "stone",
            "1": "stone_pickaxe",
            "2": "table",
            "3": "wood",
            "4": "wood_pickaxe",
        }
    )
        self.num_options = len(self.models["skills"])
        self.skills = self.models["skills"]

        # ---- Action space mapping
        self.num_primitives = int(num_primitives)
        # self.num_options = int(num_options)
        assert self.num_primitives > 0, "Need at least 1 primitive action"

        # Ensure we don't exceed underlying primitive count
        assert self.num_primitives <= self.env.action_space.n, \
            f"num_primitives={self.num_primitives} exceeds base primitive actions ({self.env.action_space.n})"

        # Hybrid action space: primitives first, then options
        self.action_space = spaces.Discrete(self.num_primitives + self.num_options)

        # Observations are passed through
        self.observation_space = self.env.observation_space

        # Book-keeping
        self.elapsed_macro_steps = 0
        self.current_obs = None  # <-- always keep latest obs for masks & BC

        # Seed same as base
        self._seed = getattr(self.env, "_seed", 0)
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

        

    # ---------- utilities ----------
    @staticmethod
    def _as_uint8_frame(obs):
        """
        Ensure obs is HxWxC uint8 (what most BC models expect).
        If obs already uint8 in [0,255], return as-is.
        If float (assumed [0,1]), scale -> uint8.
        """
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                return obs
            # assume float in [0,1]
            return (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        raise TypeError(f"Unsupported obs type for BC: {type(obs)}")

    # ---------- MaskablePPO hook ----------
    def action_masks(self):
        P, K = self.num_primitives, self.num_options
        prim_mask = np.ones(P, dtype=bool)
        if K == 0:
            return prim_mask

        # If we haven't reset yet, be conservative: only primitives are valid
        if self.current_obs is None:
            return np.concatenate([prim_mask, np.zeros(K, dtype=bool)], axis=0)

        # state -> boolean mask for all models["skills"], based on OBS
        frame = self._as_uint8_frame(self.current_obs)
        full_mask = available_skills(self.models, frame)  # aligned to models["skills"]

        # remap to current exposed subset
        skills_order = self.models["skills"]
        idx_map = {s: i for i, s in enumerate(skills_order)}
        opt_mask = np.array([full_mask[idx_map[s]] for s in self.skills], dtype=bool)
        return np.concatenate([prim_mask, opt_mask], axis=0)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.elapsed_macro_steps = 0
        self.current_obs = obs
        # reset composite progress across episodes
        if "composite_runtime" in self.models:
            self.models["composite_runtime"].clear()
        return obs, info

    def step(self, a):
        if not np.isscalar(a):
            a = int(np.asarray(a).item())
        if a < 0 or a >= self.action_space.n:
            raise error.InvalidAction(
                f"Action {a} out of range for Discrete({self.action_space.n})"
            )

        P, K = self.num_primitives, self.num_options

        # --- Case 1: primitive action -> single low-level step
        if a < P:
            obs, r, terminated, truncated, info = self.env.step(a)
            self.current_obs = obs  # update latest obs
            self.elapsed_macro_steps += 1
            return obs, float(r), bool(terminated), bool(truncated), info

        # --- Case 2: option/skill -> run BC until termination
        # print("Running option", a - P, "(", self.skills[a - P], ")")
        skill_id = a - P  # skill id in [0..K-1]
        skill_name = self.skills[skill_id]
        if "composite_runtime" in self.models:
            self.models["composite_runtime"].pop(skill_name, None)
        total_reward = 0.0
        discount = 1.0
        inner_steps = 0
        terminated = False
        truncated = False
        last_info = {}

        # start from the latest obs
        obs_local = self.current_obs
        # if obs_local is None:
        #     # print("No current observation available, resetting environment.")
        #     # If step called before reset, fall back to env.reset()
        #     obs_local, _ = self.env.reset()
        #     self.current_obs = obs_local

        
        while True:
            # print(f" Option {skill_name} step {inner_steps+1}")
            frame = self._as_uint8_frame(obs_local)
            prim_action = int(bc_policy_hierarchy(self.models, frame, skill_name))
            if prim_action < 0 or prim_action >= P:
                raise error.InvalidAction(f"bc_policy returned invalid primitive {prim_action} (P={P})")

            obs_local, r, term, trunc, info = self.env.step(prim_action)
            last_info = info
            total_reward += discount * float(r)
            discount *= self.gamma
            inner_steps += 1

            # termination checks (always via OBS)
            if term or trunc:
                terminated, truncated = bool(term), bool(trunc)
                # print(f"Option {skill_name} ended due to env done after {inner_steps} steps.")
                break
            if should_terminate(self.models, self._as_uint8_frame(obs_local), skill_name):
                # print(f"Option {skill_name} terminated after {inner_steps} steps due to model.")
                break
            if inner_steps >= self.max_skill_len:
                # print(f"Option {skill_name} reached max_skill_len={self.max_skill_len}, stopping.")
                break
        
            

        # commit the final obs from the option rollout
        self.current_obs = obs_local
        self.elapsed_macro_steps += 1
        return obs_local, float(total_reward), terminated, truncated, last_info

    # Optional passthroughs
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()
    
import numpy as np
import random
import random
import numpy as np
import imageio

import random
import numpy as np
import imageio

# --- your env imports here ---
# from craftax_env import CraftaxTopDownEnv
# from options_env import OptionsOnTopEnv

def print_options_mask(tag, mask, start, count):
    opts = mask[start:start+count]
    # print(f"{tag} options mask: {opts.tolist()}")

if __name__ == "__main__":
    # Build envs
    base = CraftaxTopDownEnv(
        render_mode="rgb_array",
        reward_items=["wood"],
        done_item="wood_pickaxe",
        include_base_reward=False,
        return_uint8=True,
    )
    env = OptionsOnTopEnv(base_env=base, num_primitives=17, num_options=5, gamma=0.99)

    # Reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Reset and capture first frame
    frames = []
    obs, info = env.reset(seed=seed)
    frames.append(obs.copy())

    # Identify option range
    num_primitives = getattr(env, "num_primitives", 17)
    num_options = getattr(env, "num_options", 5)
    option_start = num_primitives

    # Sequence of skill indices within options
    skills_seq = [0, 0, 0, 4, 4, 0, 0, 2, 2]

    # Print mask BEFORE any option
    mask = env.action_masks()
    print_options_mask("BEFORE first run", mask, option_start, num_options)

    rewards = []

    def run_skill(skill_idx, tag):
        # nonlocal obs
        action_id = option_start + skill_idx
        mask = env.action_masks()
        valid = bool(mask[action_id]) if action_id < len(mask) else False
        skill_name = None
        if hasattr(env, "skills"):
            try:
                skill_name = env.skills[skill_idx]
            except Exception:
                skill_name = None
        skill_label = f"{skill_idx}" + (f" ({skill_name})" if skill_name else "")

        # print(f"{tag}: running OPTION {skill_label} [action id {action_id}], valid={valid}")
        obs, r, terminated, truncated, info = env.step(action_id)
        frames.append(obs.copy())
        # print(f"{tag}: reward={r}, terminated={terminated}, truncated={truncated}")

        mask_after = env.action_masks()
        print_options_mask(f"AFTER {tag}", mask_after, option_start, num_options)

        if terminated or truncated:
            print(f"{tag}: episode ended → resetting")
            obs, _ = env.reset()
            frames.append(obs.copy())
        
        print("===========")  # blank line
        return r
    

    # Execute the sequence
    for i, s in enumerate(skills_seq, 1):
        r = run_skill(s, f"Run {i}")
        rewards.append(r)

    # Save GIF
    seq_str = "-".join(map(str, skills_seq))
    rewards_str = "_".join(str(float(r)) for r in rewards)  # cast to float for clean printing
    out_path = f"craftax_skills_{seq_str}_seed_{seed}_rewards_{rewards_str}.gif"
    imageio.mimsave(out_path, frames, fps=5)
    print(f"Saved GIF to: {out_path}")

# if __name__ == "__main__":
#     base = CraftaxTopDownEnv(
#         render_mode="rgb_array",
#         reward_items=["wood"],
#         done_item="stone_pickaxe",
#         include_base_reward=False,
#         return_uint8=True,
#     )
#     env = OptionsOnTopEnv(base_env=base, num_primitives=17, num_options=5, gamma=0.99)
#     all_obs = []
#     # Seed everything for reproducibility
#     seed = 0
#     random.seed(seed)
#     np.random.seed(seed)

#     obs, info = env.reset(seed=seed)
#     all_obs.append(obs.copy())

#     # Locate where options start in the mask
#     mask = env.action_masks()
#     n_actions = len(mask)
#     option_start = getattr(env, "num_primitives", 17)
#     option_end = n_actions
#     option_indices = list(range(option_start, option_end))

#     # 1) Check mask BEFORE running option 0
#     options_mask_before = mask[option_start:option_end]
#     print(f"Options mask BEFORE option 0: {options_mask_before.tolist()}")

#     # 2) Run option 0 (wood)
#     opt0 = option_start + 0
#     skill_name = env.skills[0] if hasattr(env, "skills") and len(env.skills) > 0 else "unknown"
#     print(f"Running OPTION 0 ({skill_name}) [action id {opt0}], valid={bool(mask[opt0])}")
#     obs, r, terminated, truncated, info = env.step(opt0)
#     all_obs.append(obs.copy())
#     print(f"Option 0 result -> Reward: {r} | Terminated: {terminated} | Truncated: {truncated}")

#     # 3) Check mask AFTER running option 0
#     mask = env.action_masks()
#     options_mask_after_opt = mask[option_start:option_end]
#     print(f"Options mask AFTER option 0: {options_mask_after_opt.tolist()}")

#     # If episode ended, reset before primitive steps
#     if terminated or truncated:
#         print("\nEpisode ended after option. Resetting before primitive steps...\n")
#         obs, info = env.reset()

#     # 4) Run primitive action 0 for 5 steps
#     steps_to_run = 5
#     for i in range(steps_to_run):
#         obs, r, terminated, truncated, info = env.step(0)
#         all_obs.append(obs.copy())
#         print(f"Primitive step {i+1}/{steps_to_run} -> Reward: {r} | Terminated: {terminated} | Truncated: {truncated}")
#         if terminated or truncated:
#             print("\nEpisode ended during primitive steps. Resetting...\n")
#             obs, info = env.reset()
#             break

#     # 5) Check mask AFTER primitive steps
#     mask = env.action_masks()
#     options_mask_after_prim = mask[option_start:option_end]
#     print(f"Options mask AFTER {steps_to_run} primitive steps: {options_mask_after_prim.tolist()}")


#     import imageio
#     frames = [f for f in all_obs]
#     imageio.mimsave(f"craftax_run_test_seed_{seed}_{r}.gif", frames, fps=5)