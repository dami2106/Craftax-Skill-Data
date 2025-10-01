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
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error

from top_down_env_gymnasium import CraftaxTopDownEnv

# IMPORTANT: import the updated helpers
from option_helpers import (
    available_skills,
    bc_policy_hierarchy,
    load_all_models_hierarchy,
    new_call_id,
    should_terminate,          # now phase/instance-aware
)

class OptionsOnTopEnv(gym.Env):
    """
    Wraps CraftaxTopDownEnv to expose a Discrete action space:
      [0..P-1]      -> primitive actions (single step)
      [P..P+K-1]    -> options (macro-step via BC until termination)

    Primitives are always valid; options are valid iff available_skills(obs)[i] is True.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(
        self,
        base_env,
        num_primitives: int = 17,
        gamma: float = 0.99,
        max_skill_len: int = 100,
    ):
        super().__init__()
        self.env = base_env
        self.gamma = float(gamma)
        self.max_skill_len = int(max_skill_len)

        # Models / skills (now come with instance-scoped runtime support)
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
        assert self.num_primitives > 0, "Need at least 1 primitive action"
        assert self.num_primitives <= self.env.action_space.n, \
            f"num_primitives={self.num_primitives} exceeds base primitive actions ({self.env.action_space.n})"

        self.action_space = spaces.Discrete(self.num_primitives + self.num_options)
        self.observation_space = self.env.observation_space

        self.elapsed_macro_steps = 0
        self.current_obs = None

        self._seed = getattr(self.env, "_seed", 0)
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

        self.debug_record_frames = None


    def _skill_budget(self, skill_name: str) -> int:
        """
        Primitive skills get max_skill_len.
        Composite skills get (#leaf_skills) * max_skill_len.
        """
        entry = self.models["bc_models"].get(skill_name)
        if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
            seq = entry[1]             # list of leaf skill names
            return len(seq) * self.max_skill_len
        return self.max_skill_len
    
    # ---------- utilities ----------
    @staticmethod
    def _as_uint8_frame(obs):
        """
        Ensure obs is HxWxC uint8 (what most BC models expect).
        """
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                return obs
            return (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        raise TypeError(f"Unsupported obs type for BC: {type(obs)}")

    # ---------- MaskablePPO hook ----------
    def action_masks(self):
        P, K = self.num_primitives, self.num_options
        prim_mask = np.ones(P, dtype=bool)
        if K == 0:
            return prim_mask

        if self.current_obs is None:
            return np.concatenate([prim_mask, np.zeros(K, dtype=bool)], axis=0)

        frame = self._as_uint8_frame(self.current_obs)
        full_mask = available_skills(self.models, frame)  # aligned to models["skills"]

        skills_order = self.models["skills"]
        idx_map = {s: i for i, s in enumerate(skills_order)}
        opt_mask = np.array([full_mask[idx_map[s]] for s in self.skills], dtype=bool)
        return np.concatenate([prim_mask, opt_mask], axis=0)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.elapsed_macro_steps = 0
        self.current_obs = obs
        # Clear *all* composite instance state across episodes
        if "composite_runtime" in self.models:
            self.models["composite_runtime"].clear()
        # also reset counter to keep logs tidy (optional)
        if "call_id_ctr" in self.models:
            self.models["call_id_ctr"] = 0
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
            self.current_obs = obs
            self.elapsed_macro_steps += 1
            return obs, float(r), bool(terminated), bool(truncated), info

        # --- Case 2: option/skill -> run BC until termination (instance-scoped)
        skill_id = a - P
        skill_name = self.skills[skill_id]

        # New instance id for this macro execution
        call_id = new_call_id(self.models)

        total_reward = 0.0
        discount = 1.0
        inner_steps = 0
        terminated = False
        truncated = False
        last_info = {}

        obs_local = self.current_obs

        step_budget = self._skill_budget(skill_name)

        while True:
            frame = self._as_uint8_frame(obs_local)
            prim_action = int(
                bc_policy_hierarchy(
                    self.models, frame, skill_name, call_id,
                    max_leaf_len=self.max_skill_len   # <- per-leaf cap
                )
            )
            if prim_action < 0 or prim_action >= P:
                raise error.InvalidAction(f"bc_policy returned invalid primitive {prim_action} (P={P})")

            obs_local, r, term, trunc, info = self.env.step(prim_action)
            last_info = info
            total_reward += discount * float(r)
            discount *= self.gamma
            inner_steps += 1

            if self.debug_record_frames is not None:
                self.debug_record_frames.append(obs_local.copy())

            # termination checks
            if term or trunc:
                terminated, truncated = bool(term), bool(trunc)
                break

            if should_terminate(self.models, self._as_uint8_frame(obs_local), skill_name, call_id):
                # Composite termination is now phase-aware: only ends on final leaf
                break

            if inner_steps >= step_budget:
                break

        # commit final obs
        self.current_obs = obs_local
        self.elapsed_macro_steps += 1
        return obs_local, float(total_reward), terminated, truncated, last_info

    # Optional passthroughs
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()


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
        reward_items=[],
        done_item="wood_pickaxe",
        include_base_reward=False,
        return_uint8=True,
    )
    env = OptionsOnTopEnv(base_env=base, num_primitives=16, gamma=0.99, max_skill_len=25)
    # env.max_skill_len = 30  
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
    # Available skills: ['wood' , 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table', 'Production_0', 'Production_1', 'Production_8 WTW 23', 'Production_10', 'Production_9', 'Production_14']


    skills_seq = [ 23] # w w w 



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
    # Annotate each frame with the frame number in the top left
    annotated_frames = []
    for idx, frame in enumerate(frames):
        # Ensure frame is in uint8 format
        img = frame.copy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        # Add text
        cv2.putText(
            img,
            f"Frame: {idx}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated_frames.append(img)

    seq_str = "-".join(map(str, skills_seq))
    rewards_str = "_".join(str(float(r)) for r in rewards)
    out_path = f"craftax_seq_.gif"
    imageio.mimsave(out_path, annotated_frames, fps=10)
    print(f"Saved GIF to: {out_path}")