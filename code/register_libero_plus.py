"""Register the LIBERO-plus RLDS training set with OpenVLA-OFT's OXE data pipeline.

Runtime registration only -- the official repo at /opt/vla-retain/third_party/openvla-oft is left
byte-for-byte untouched. Import this module (and call `register()`) before RLDSDataset is built.

WHY NO NEW TRANSFORM IS NEEDED
------------------------------
The published LIBERO-plus RLDS has exactly the schema OFT's stock `libero_dataset_transform`
already assumes (verified against features.json of the shipped dataset):

    steps/observation/image        (256, 256, 3) uint8
    steps/observation/wrist_image  (256, 256, 3) uint8
    steps/observation/state        (8,)  float32   -> EEF_state = state[:6], gripper = state[-2:]
    steps/observation/joint_state  (7,)  float32   (unused by the OFT LIBERO recipe)
    steps/action                   (7,)  float32   -> 6 EEF deltas + 1 gripper
    steps/language_instruction     string

so the stock transform applies unchanged: it slices state[:, :6] / state[:, -2:] and clips+inverts
the last action channel into the gripper convention OFT expects. Writing our own transform here
would only risk diverging from the convention the released OFT+ checkpoint was trained under.
"""

from prismatic.vla.datasets.rlds.oxe.configs import (
    ActionEncoding,
    OXE_DATASET_CONFIGS,
    StateEncoding,
)
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
    libero_dataset_transform,
)

# TFDS name of the published training set: <data_root_dir>/libero_mix/1.0.0/
DATASET_NAME = "libero_mix"


def register(verbose: bool = True) -> str:
    """Insert LIBERO-plus into the three registries OFT's loader consults. Idempotent."""
    OXE_DATASET_CONFIGS[DATASET_NAME] = {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }
    OXE_STANDARDIZATION_TRANSFORMS[DATASET_NAME] = libero_dataset_transform
    OXE_NAMED_MIXTURES[DATASET_NAME] = [(DATASET_NAME, 1.0)]

    if verbose:
        print(f"[register] {DATASET_NAME}: config + transform + mixture registered", flush=True)
    return DATASET_NAME
