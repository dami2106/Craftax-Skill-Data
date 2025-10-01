# options_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error

from top_down_env_gymnasium import CraftaxTopDownEnv

from option_helpers import *

import numpy as np
import random
import random
import numpy as np
import imageio

import random
import numpy as np
import imageio

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
        num_primitives: int = 16,    
        gamma: float = 0.99,
        max_skill_len: int = 50,      # safety cap for option rollout
    ):
        super().__init__()
        self.env = base_env
        self.gamma = float(gamma)
        self.max_skill_len = int(max_skill_len)

        # Models / skills
        self.models = load_all_models()
        self.skills = self.models["skills"]

        # ---- Action space mapping
        self.num_primitives = int(num_primitives)
        self.num_options = int(len(self.skills)) if self.skills is not None else 0
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

        self.debug_record_frames = None

        self.active_option = None

        

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

        if self.current_obs is None:
            return np.concatenate([prim_mask, np.zeros(K, dtype=bool)])

        if self.active_option is not None:
            # Option is currently running; you can constrain choices.
            # Example: only allow “continue same option” and primitives.
            skills_order = self.models["skills"]
            idx_map = {s: i for i, s in enumerate(skills_order)}
            opt_mask = np.zeros(K, dtype=bool)
            opt_mask[[self.skills.index(self.active_option)]] = True

            # print("Active option mask:", opt_mask)

            return np.concatenate([prim_mask, opt_mask])

        # No active option -> compute availability as you already do
        frame = self._as_uint8_frame(self.current_obs)
        full_mask = available_skills(self.models, frame)
        skills_order = self.models["skills"]
        idx_map = {s: i for i, s in enumerate(skills_order)}
        opt_mask = np.array([full_mask[idx_map[s]] for s in self.skills], dtype=bool)
        return np.concatenate([prim_mask, opt_mask])

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.elapsed_macro_steps = 0
        self.current_obs = obs  # keep latest obs
        return obs, info

    def step(self, a):
        if not np.isscalar(a):
            a = int(np.asarray(a).item())
        if a < 0 or a >= self.action_space.n:
            raise error.InvalidAction(
                f"Action {a} out of range for Discrete({self.action_space.n})"
            )

        P, K = self.num_primitives, self.num_options

        # print("Step called with active option:", self.active_option)

        # Determine which option (if any) should be active for *this* step
        if a < P:
            # agent picked a primitive -> clear any active option and execute that primitive

            # print("Picked primitive action:", a, "clearing active option.")

            self.active_option = None
            prim_action = a

        else:
            # agent picked an option
            # print("Picked option :", a)
            picked_skill_id = a - P
            picked_skill = self.skills[picked_skill_id]
            # set or keep active option
            self.active_option = picked_skill
            # compute exactly ONE primitive to execute this step via BC:
            frame = self._as_uint8_frame(self.current_obs)
            prim_action = int(bc_policy(self.models, frame, self.active_option))
            # (validate prim_action in [0, P-1])

        # ---- one primitive step only ----
        obs, r, terminated, truncated, info = self.env.step(prim_action)
        self.current_obs = obs
        self.elapsed_macro_steps += 1  # optional bookkeeping

        # Option termination check (runs every step while active)
        if self.active_option is not None:
            if terminated or truncated or should_terminate(self.models, self._as_uint8_frame(obs), self.active_option):
                # print("Option", self.active_option, "terminated due to end ep or model.")
                self.active_option = None

        return obs, float(r), bool(terminated), bool(truncated), info

    # Optional passthroughs
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()
    


def print_options_mask(tag, mask, start, count):
    opts = mask[start:start+count]
    # print(f"{tag} options mask: {opts.tolist()}")

if __name__ == "__main__":
    # Build envs
    base = CraftaxTopDownEnv(
        render_mode="rgb_array",
        reward_items=[],
        done_item="wood_pickaxe",
        include_base_reward=False,
        return_uint8=True,
    )
    env = OptionsOnTopEnv(base_env=base, num_primitives=16, gamma=0.99)
    env.max_skill_len = 30  
    # env.max_skill_len = 20     


    # Reproducibility
    seed = 888
    random.seed(seed)
    np.random.seed(seed)

    # Reset and capture first frame
    frames = []
    env.debug_record_frames = frames
    obs, info = env.reset(seed=seed)
    frames.append(obs.copy())

    num_primitives = env.num_primitives
    num_options = env.num_options
    print(f"Env with {num_primitives} primitives + {num_options} options (total {env.action_space.n})")
    print("Available skills:", env.skills)

    # Now this list can contain both primitives and options directly
    # Available skills: ['wood',    'stone',     'wood_pickaxe',     'stone_pickaxe',      'table']
    # Available skills: ['16',      '17',        '18',               '19',                 '20']
    # e.g. [0, 1, 17, 20] means: primitive 0, primitive 1, option 1, option 4
    skills_seq = [16, 16, 16, 5, 20, 18] # w w w 
    # Print mask BEFORE any run
    mask = env.action_masks()
    print_options_mask("BEFORE first run", mask, num_primitives, num_options)

    rewards = []

    def run_action(a, tag):
        mask = env.action_masks()
        valid = bool(mask[a]) if a < len(mask) else False
        label = str(a)

        if a >= num_primitives and hasattr(env, "skills"):
            skill_idx = a - num_primitives
            if 0 <= skill_idx < len(env.skills):
                label += f" (option:{env.skills[skill_idx]})"

        print(f"{tag}: running action {label}, valid={valid}")
        obs, r, terminated, truncated, info = env.step(a)
        frames.append(obs.copy())
        print(f"{tag}: reward={r}, terminated={terminated}, truncated={truncated}")

        mask_after = env.action_masks()
        print_options_mask(f"AFTER {tag}", mask_after, num_primitives, num_options)

        # if terminated or truncated:
        #     print(f"{tag}: episode ended → resetting")
        #     obs, _ = env.reset()
        #     frames.append(obs.copy())

        print("===========")
        return r

    # Execute the sequence (mix primitives + options)
    for i, a in enumerate(skills_seq, 1):
        r = run_action(a, f"Run {i}")
        rewards.append(r)

    # Save GIF
    seq_str = "-".join(map(str, skills_seq))
    rewards_str = "_".join(str(float(r)) for r in rewards)
    out_path = f"craftax_seq_.gif"
    imageio.mimsave(out_path, frames, fps=10)
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