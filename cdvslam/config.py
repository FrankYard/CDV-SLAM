from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions?
_C.CENTROID_SEL_STRAT = 'RANDOM'
_C.CENTROID_SAMPLE_STRAT = ''

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3 # TODO: Is is hard coded in _repetition_check?
_C.LOOP_RETR_THRESH = 0.04

# Semantic and dynamic SLAM
_C.SEG_HEAD = None
_C.FILTER_DYNAMIC_CLASS = False
_C.VIEW_FILTERED_WEIGHT = False

# Monocular depth estimation
_C.DEPTH_HEAD = None

# motion probe
_C.MOTION_PROBE_THR = 2.0

cfg = _C
