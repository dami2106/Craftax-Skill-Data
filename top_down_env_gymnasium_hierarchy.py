# options_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error
from top_down_env_gymnasium import CraftaxTopDownEnv

# IMPORTANT: import the updated helpers
from option_helpers import (
    bc_policy_hierarchy,
    load_all_models_hierarchy,
    new_call_id,
    _clear_rt,
)

class OptionsOnTopEnv(gym.Env):
    """
    Discrete action space:
      [0..P-1]      -> primitive actions (single step)
      [P..P+K-1]    -> options (macro policies via BC, executed call-and-return)

    SMDP semantics:
      - Agent selects either a primitive (1 env step) or an option.
      - If an option is chosen, the wrapper internally rolls out primitives for
        up to a skill-specific budget, accumulating rewards before returning.
      - Masks only guard the selection step (all primitives/options currently valid).
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(
        self,
        base_env,
        num_primitives: int = 17,
        gamma: float = 0.99,
        max_skill_len: int = 100,   # per-leaf cap; composites scale by number of leaves

        skill_list=['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
        hierarchies_dir='Traces/stone_pickaxe_easy/hierarchy_data/Simple',
        symbol_map={
            "0": "stone",
            "1": "stone_pickaxe",
            "2": "table",
            "3": "wood",
            "4": "wood_pickaxe",
        },
        root: str = 'Traces/stone_pickaxe_easy',
        backbone_hint:  str = 'resnet34',
        bc_checkpoint_dir: str = 'bc_checkpoints_resnet',
        dataset_mean_std_path: str = 'dataset_mean_std.npy',
        pu_start_models_dir: str = 'pu_start_models',
        pu_end_models_dir: str = 'pu_end_models',

    ):
        super().__init__()
        self.env = base_env
        self.gamma = float(gamma)
        self.max_skill_len = int(max_skill_len)


        # Models / skills (supports composite skills + instance-scoped runtime)
        self.models = load_all_models_hierarchy(
            skill_list,
            hierarchies_dir,
            symbol_map,
            root,
            backbone_hint,
            bc_checkpoint_dir,
            dataset_mean_std_path,
            pu_start_models_dir,
            pu_end_models_dir,
        )

        self.skills = self.models["skills"]
        self.num_options = len(self.skills)

        print("Number of options:", self.num_options)
        print("Available skills:", self.skills)
        print("symbol_map:", symbol_map)
        print("Available skills (full):", self.models["skills"])

        # ---- Action space mapping
        self.num_primitives = int(num_primitives)
        assert self.num_primitives > 0, "Need at least 1 primitive action"
        assert self.num_primitives <= self.env.action_space.n, \
            f"num_primitives={self.num_primitives} exceeds base primitive actions ({self.env.action_space.n})"

        self.action_space = spaces.Discrete(self.num_primitives + self.num_options)
        self.observation_space = self.env.observation_space

        # Book-keeping
        self.current_obs = None
        self.total_primitive_steps = 0

        # Seeding consistent with base
        self._seed = getattr(self.env, "_seed", 0)
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

    # ---------- utilities ----------
    def _skill_budget(self, skill_name: str) -> int:
        """
        Primitive skills get max_skill_len.
        Composite skills get (#leaf_skills) * max_skill_len.
        We infer composite by checking models["bc_models"][skill_name] == ("__COMPOSITE__", seq).
        """
        entry = self.models["bc_models"].get(skill_name)
        if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
            seq = entry[1]             # list of leaf skill names
            return len(seq) * self.max_skill_len
        return self.max_skill_len

    @staticmethod
    def _as_uint8_frame(obs):
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                return obs
            return (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        raise TypeError(f"Unsupported obs type for BC: {type(obs)}")

    # ---------- MaskablePPO hook ----------
    def action_masks(self):
        P, K = self.num_primitives, self.num_options
        if K == 0:
            return np.ones(P, dtype=bool)

        return np.ones(P + K, dtype=bool)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.current_obs = obs
        self.total_primitive_steps = 0

        # Clear composite/instance runtime across episodes
        if "composite_runtime" in self.models:
            self.models["composite_runtime"].clear()
        if "call_id_ctr" in self.models:
            self.models["call_id_ctr"] = 0

        return obs, info

    def step(self, a):
        # Normalize scalar action
        if not np.isscalar(a):
            a = int(np.asarray(a).item())
        if a < 0 or a >= self.action_space.n:
            raise error.InvalidAction(f"Action {a} out of range for Discrete({self.action_space.n})")

        P, K = self.num_primitives, self.num_options
        if a < P:
            return self._run_primitive(a)

        option_idx = a - P
        if option_idx < 0 or option_idx >= K:
            raise error.InvalidAction(f"Invalid option id {option_idx}")

        return self._run_option(option_idx)

    def _run_primitive(self, prim_action: int):
        obs, r, terminated, truncated, info = self.env.step(prim_action)
        self.current_obs = obs
        self.total_primitive_steps += 1
        info = dict(info) if info is not None else {}
        info["primitive_steps"] = self.total_primitive_steps
        return obs, float(r), bool(terminated), bool(truncated), info

    def _run_option(self, option_idx: int):
        skill_name = self.skills[option_idx]
        budget = self._skill_budget(skill_name)
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = {}
        steps = 0
        obs = self.current_obs

        if obs is None:
            raise RuntimeError("Environment must be reset before running options.")

        call_id = new_call_id(self.models)
        try:
            while steps < budget:
                frame = self._as_uint8_frame(obs)
                prim_action = self._bc_one_step_for_option(skill_name, frame, call_id)
                obs, r, terminated, truncated, last_info = self.env.step(prim_action)

                total_reward += float(r)
                self.total_primitive_steps += 1
                steps += 1
                self.current_obs = obs

                if terminated or truncated:
                    break
        finally:
            _clear_rt(self.models, skill_name, call_id)

        if last_info is None:
            last_info = {}
        info = dict(last_info)
        option_info = info.setdefault("option", {})
        option_info.update(
            {
                "skill": skill_name,
                "steps": steps,
                "reward": total_reward,
                "budget": budget,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if steps >= budget and not (terminated or truncated):
            option_info["budget_exhausted"] = True

        info["primitive_steps"] = self.total_primitive_steps

        return obs, total_reward, bool(terminated), bool(truncated), info

    def _bc_one_step_for_option(self, skill_name: str, frame_uint8: np.ndarray, call_id: int) -> int:
        prim = int(
            bc_policy_hierarchy(
                self.models,
                frame_uint8,
                skill_name,
                call_id,
                max_leaf_len=self.max_skill_len,
            )
        )
        if prim < 0 or prim >= self.num_primitives:
            raise error.InvalidAction(
                f"bc_policy returned invalid primitive {prim} (P={self.num_primitives})"
            )
        return prim

    # Optional passthroughs
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

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