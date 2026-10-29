# BFieldPINN
Development for physics-informed neural network (PINN) to aid in modeling Mu2e magnetic field.

# Installation
1. Create an anaconda environemnt called `mu2eBFit` using the environment file in the first-level directory in the repository. This environment can be used to run the PINN and LSQ fits. Note if you've already created the environment for the LSQ fit, start at the command `conda activate mu2eBFit`.
```
(base) ~/FMS_BField_Model/$ conda env create -f environment_mu2eBFit.yml
```

2. Activate the conda environment:
```
(base) ~/FMS_BField_Model$ conda activate mu2eBFit
(mu2eBFit) ~/FMS_BField_Model$
```

3. Navigate to `BFieldPINN_package` and create a symbolic link to the output data directory (replace "YOUR_DATA_DIR" with the directory on your computer where you want to store results):
```
(mu2eBFit) ~/FMS_BField_Model$ cd BFieldPINN_package
(mu2eBFit) ~/FMS_BField_Model/BFieldPINN_package/$ ln -s YOUR_DATA_DIR data
```

4. Install the `BFieldPINN` package in the conda environment:
```
(mu2eBFit) ~/FMS_BField_Model/BFieldPINN_package/$ pip install -e .
```

# Reproducibility
By default PINN training is **not** bit-reproducible: unseeded draws differ from run to run, and several TensorFlow GPU kernels sum in a
non-deterministic order, so two runs with identical seeds already differ in the first epoch. That spread is real and worth quoting
(train several seeds). When a study needs identical results from identical inputs, fix every random draw and turn on TF op determinism:

| draw | controlled by |
|---|---|
| train / validation split | numpy global seed (`np.random.seed`, set by the driver) |
| input + hidden layer initial weights | `NN_dict['initializer_seed']` (`tools.prep_PINN_inputs`); biases start at zero |
| output layer initial weights | Keras default `glorot_uniform`, unseeded: its seed comes from Python's `random` |
| collocation points | `NN_dict['colloc_seed']` (three draws use seed, seed + 1, seed + 2; `None` = unseeded) |
| GPU kernel summation order | `tf.config.experimental.enable_op_determinism()` |

`tools.set_determinism(tf_seed, np_seed=None, py_seed=None)` does the global part in one call (TF global seed, Python `random`, numpy if
given, op determinism); set `initializer_seed` and `colloc_seed` in the NN config. Call it once, before any data is drawn or the PINN is built.

Notes:
- Results are bit-identical only on the same GPU model, driver, CUDA / cuDNN and TensorFlow versions. Across machines, compare within the
  seed spread instead.
- TensorFlow raises an error if an op has no deterministic GPU implementation (none in the current `ScalarPINN` training).
- Verified 2026-10-28 (docdb-48750, model 1, RTX 2080 Ti, TF 2.14), 200-epoch runs: bit-identical in every per-epoch loss, all initial and
  trained weights and the predictions; full model-1 chains (PINN, LSQ refit, merge) on two different cards of the same model: bit-identical in all of
  these and in the final refit + PINN results; no measurable slowdown (1491 s vs ~1500-1650 s unseeded).
- With `initializer_seed` set, one initializer instance is reused for every hidden layer and Keras returns the same values on each call, so the
  hidden weight matrices start identical (as they also do unseeded). This is the current behaviour of every model; changing it is an open item.
