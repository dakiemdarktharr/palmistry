"""NumPy pose regression and parametric palm-line tracing, independent of mask schemas."""
SCHEMA = 'palm-keypoints-v1'
LINES = ('heart_line', 'head_line', 'life_line', 'fate_line')
N_POINTS = 6
TOTAL_POINTS = 26  # 4 x 6 line landmarks + 2 palm-width landmarks
