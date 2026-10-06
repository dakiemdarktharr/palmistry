"""NumPy pose regression and parametric palm-line tracing, independent of mask schemas."""
SCHEMA = 'palm-keypoints-v2'
LEGACY_SCHEMA = 'palm-keypoints-v1'
LINES = ('heart_line', 'head_line', 'life_line')
N_POINTS = 6
LINE_POINTS = len(LINES) * N_POINTS
TOTAL_POINTS = LINE_POINTS + 2
COORDS = TOTAL_POINTS * 2
