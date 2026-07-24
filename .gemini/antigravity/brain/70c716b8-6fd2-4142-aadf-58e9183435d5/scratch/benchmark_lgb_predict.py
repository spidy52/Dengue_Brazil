import lightgbm as lgb
import numpy as np
import time

# Create dummy training data
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

train_data = lgb.Dataset(X, label=y)
params = {"objective": "regression", "verbose": -1}
model = lgb.train(params, train_data, num_boost_round=50)

# Benchmark single-row predictions
t0 = time.time()
n_trials = 10000
for i in range(n_trials):
    x_test = X[i % 1000 : (i % 1000) + 1, :]
    pred = model.predict(x_test)
duration = time.time() - t0
print(f"Time for {n_trials} single-row predictions: {duration:.4f} seconds")
print(f"Average time per prediction: {duration/n_trials*1000:.4f} ms")
print(f"Estimated time for 1.45M predictions: {duration/n_trials * 1450000 / 60:.2f} minutes")
