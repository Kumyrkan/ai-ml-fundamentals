# Wine PCA Analysis and Matrix Puzzles

This project consists of two parts: the implementation of vectorized matrix operations and a from-scratch Principal Component Analysis (PCA) of the UCI Wine dataset using only NumPy.

## Matrix Puzzles
We implemented five core matrix routines (`rotate_90`, `transpose_no_T`, `is_magic_square`, `block_trace`, and `top_k_indices`) using pure NumPy vectorization. By avoiding Python for-loops, we ensured optimal computational efficiency. The implementation was verified with a suite of 23 pytest cases, covering basic functionality, edge cases, and invariant properties, all of which passed successfully.

## Wine PCA Analysis
The analysis of the UCI Wine dataset involved standardizing 13 chemical features and decomposing their variance. Our PCA implementation yielded the following results:
- **80% Variance:** To explain at least 80% of the total variance, **5 principal components** are required (cumulative variance: 80.16%).
- **95% Variance:** To capture at least 95% of the information, **10 principal components** are necessary (cumulative variance: 96.17%).

The projection of the dataset onto the first two principal components shows that the three wine cultivars are visibly separated in the 2D space. This is confirmed by the distinct means of the projections for each class:
- **Class 1 mean:** [-2.2827, 0.9679]
- **Class 2 mean:** [0.0390, -1.6435]
- **Class 3 mean:** [2.7482, 1.2413]

The clear distance between these centroids demonstrates that PCA effectively reduces the dimensionality of the chemical data while preserving enough information to distinguish between the wine varieties with high confidence.