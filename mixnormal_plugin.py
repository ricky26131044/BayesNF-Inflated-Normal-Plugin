# mixnormal_plugin.py
# 目的：不改 bayesnf 原始碼，新增 observation_model = "MIXTURE_NORMAL"
#      兩常態混和，共用均值 F(s,t)，並預測 p(s,t)。
#
# 使用方式：
#   from mixnormal_plugin import enable_mixture_normal, predict_F_p
#   enable_mixture_normal()
#   from bayesnf.spatiotemporal import BayesianNeuralFieldMAP
#   est = BayesianNeuralFieldMAP(..., observation_model="MIXTURE_NORMAL")
#   est.fit(...)
#   F_samples, p_samples, F_mean, p_mean = predict_F_p(est, table)

from __future__ import annotations

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

_PATCHED = False

import enum

class LikelihoodDistPatched(enum.Enum):
    NORMAL = "NORMAL"
    NB = "NB"
    ZINB = "ZINB"
    MIXTURE_NORMAL = "MIXTURE_NORMAL"      # 舊名保留當 alias
    INFLATED_NORMAL = "INFLATED_NORMAL"    # 新名（點膨脹 + 高斯）

class BayesianNeuralField1D_Mixture(nn.Module):
    """BayesNF 架構的 2-head 版本：輸出 [F, logit_p]"""

    width: int
    depth: int
    input_scales: np.ndarray
    fourier_degrees: np.ndarray
    interactions: np.ndarray
    num_seasonal_harmonics: np.ndarray = flax.struct.field(
        default_factory=lambda: np.zeros((0,))
    )
    seasonality_periods: np.ndarray = flax.struct.field(
        default_factory=lambda: np.zeros((0,))
    )

    @nn.compact
    def __call__(self, x):
        # 盡量沿用 bayesnf.models 的 feature functions
        import bayesnf.models as models

        init = nn.initializers.normal(1.0)

        if len(x.shape) == 1:
            x = x[..., jnp.newaxis]
        log_scale_adjustment = self.param("log_scale_adjustment", init, x.shape[-1:])
        scaled_x = x / (self.input_scales * jnp.exp(log_scale_adjustment))

        seasonal_features = models.make_seasonal_features(
            x[..., 0],
            self.seasonality_periods,
            self.num_seasonal_harmonics,
            rescale=True,
        )

        fourier_features = [
            models.make_fourier_features(scaled_x[..., i], degree, rescale=True)
            for i, degree in enumerate(self.fourier_degrees)
            if degree > 0
        ]

        interaction_features = jnp.prod(scaled_x[:, self.interactions], axis=-1)

        def make_layer_scale(name, shape=()):
            inv_sp = self.param(name, init, shape)
            return jax.nn.softplus(inv_sp)

        features = [scaled_x, *fourier_features, seasonal_features, interaction_features]
        features = [
            f * jax.nn.softplus(self.param(f"feature_inv_sp_scale{i}", init, ()))
            for i, f in enumerate(features)
            if f.size > 0
        ]
        h = jnp.concatenate(features, -1)

        activation_weight = jax.nn.sigmoid(self.param("logit_activation_weight", init, ()))

        def activation_fn(z):
            return activation_weight * nn.elu(z) + (1 - activation_weight) * nn.tanh(z)

        # for layer_id in range(self.depth):
        #     layer = nn.Dense(self.width, kernel_init=init, bias_init=init)
        #     layer_scale = make_layer_scale(f"inv_sp_layer_scale{layer_id}")
        #     h = h / jnp.sqrt(jnp.shape(h)[-1])
        #     h = activation_fn(layer_scale * layer(h))

        # # 2 heads: [F, logit_p]
        # output_layer = nn.Dense(2, kernel_init=init, bias_init=init)
        # output_scale = make_layer_scale("inv_sp_output_scale", (2,))

        # h = h / jnp.sqrt(h.shape[-1])
        # out = output_layer(h)  # (batch, 2)
        # return out * output_scale

        # 共享主幹：前 depth-1 層
        for layer_id in range(self.depth - 1):
            layer = nn.Dense(self.width, kernel_init=init, bias_init=init)
            layer_scale = make_layer_scale(f'inv_sp_layer_scale{layer_id}')
            h = h / jnp.sqrt(jnp.shape(h)[-1])
            h = activation_fn(layer_scale * layer(h))

        # 分叉：各自的最後一層
        h_norm = h / jnp.sqrt(h.shape[-1])
        
        # F 頭
        head_F  = nn.Dense(self.width, kernel_init=init, bias_init=init)
        scale_F = make_layer_scale('inv_sp_head_F_scale')
        h_F     = activation_fn(scale_F * head_F(h_norm))
        h_F     = h_F / jnp.sqrt(h_F.shape[-1])
        out_F   = nn.Dense(1, kernel_init=init, bias_init=init)(h_F)

        # p 頭（較淺，直接接輸出）
        head_p  = nn.Dense(1, kernel_init=init, bias_init=init)
        out_p   = head_p(h_norm)

        out_F_scaled = out_F[..., 0] * make_layer_scale('inv_sp_output_F')
        out_p_scaled = out_p[..., 0] * make_layer_scale('inv_sp_output_p')

        return jnp.stack([out_F_scaled, out_p_scaled], axis=-1)
  



# def _make_likelihood_mixture(params, x, mlp, mlp_template) -> tfd.Distribution:
#     """distribution == MIXTURE_NORMAL 時走這裡
#     params[0]=log_sigma_small
#     params[1]=delta_raw (softplus 後保證 sigma_large > sigma_small)
#     params[2]=mixing_bias
#     mlp(x) 輸出 [F, logit_p]
#     """
#     treedef = jax.tree_util.tree_structure(mlp_template)
#     mlp_params = jax.tree_util.tree_unflatten(treedef, params[3:])

#     preds = mlp.apply(mlp_params, x)  # (..., n_obs, 2) after vmap/pmap
#     F = preds[..., 0]
#     logit_p = preds[..., 1]

#     log_sigma_small = params[0]
#     delta = jax.nn.softplus(params[1])
#     log_sigma_large = log_sigma_small + delta

#     sigma_small = 0.01 + jnp.exp(log_sigma_small)
#     sigma_large = 0.01 + jnp.exp(log_sigma_large)

#     mixing_bias = params[2]
#     # broadcast 到 observation 軸
#     p = jax.nn.sigmoid(logit_p + mixing_bias[..., jnp.newaxis])
#     p = jnp.clip(p, 1e-5, 1 - 1e-5)

#     loc = jnp.stack([F, F], axis=-1)
#     scale = jnp.stack(
#         [
#             sigma_small[..., jnp.newaxis] * jnp.ones_like(F),
#             sigma_large[..., jnp.newaxis] * jnp.ones_like(F),
#         ],
#         axis=-1,
#     )

#     components = tfd.Normal(loc=loc, scale=scale)
#     mix = tfd.Categorical(probs=jnp.stack([p, 1 - p], axis=-1))
#     mixture = tfd.MixtureSameFamily(mix, components)
#     return tfd.Independent(mixture, 1)

# 你要的膨脹點（單一膨脹處 c）
INFLATED_LOC = None  # 這就是你說的「一個變數代表最小值」

def set_inflated_loc(v):
    global INFLATED_LOC
    INFLATED_LOC = float(v)

# 近似點質量的極小變異（越小越像點，但太小可能數值不穩）
INFLATED_SCALE = 1e-3

def _make_likelihood_inflated_normal(params, x, mlp, mlp_template):
    """
    y ~  {  point at c (approx by tiny Normal)  with prob p(s,t)
          Normal(F(s,t), sigma)                 with prob 1-p(s,t) }

    params[0] : log_sigma (沿用 bayesnf NORMAL 的 log_noise_scale)
    params[2] : mixing_bias (沿用 bayesnf 原本第三個 latent)
    mlp(x)    : outputs [F, logit_p]
    """
    treedef = jax.tree_util.tree_structure(mlp_template)
    mlp_params = jax.tree_util.tree_unflatten(treedef, params[3:])

    preds = mlp.apply(mlp_params, x)  # (..., n_obs, 2)
    F = preds[..., 0]                 # (..., n_obs)
    logit_p = preds[..., 1]           # (..., n_obs)

    # Gaussian part sigma
    log_sigma = params[0]
    sigma = 0.01 + jnp.exp(log_sigma)

    # p(s,t): inflation probability
    mixing_bias = params[2][..., jnp.newaxis]
    p = jax.nn.sigmoid(logit_p + mixing_bias)
    p = jnp.clip(p, 1e-5, 1 - 1e-5)

    # Inflation component: near-point mass at c
    c = INFLATED_LOC * jnp.ones_like(F)

    inflated = tfd.Normal(loc=c, scale=INFLATED_SCALE)
    # 如果你想要真正的點質量（但 quantile/root finding 可能更容易卡）
    # inflated = tfd.Deterministic(loc=c)

    gaussian = tfd.Normal(loc=F, scale=sigma[..., jnp.newaxis] * jnp.ones_like(F))

    mix = tfd.Categorical(probs=jnp.stack([p, 1 - p], axis=-1))
    mixture = tfd.Mixture(cat=mix, components=[inflated, gaussian])

    return tfd.Independent(mixture, 1)

# def enable_mixture_normal():
#     """執行一次即可，會在 runtime 內把 bayesnf 的 hook 接上 MIXTURE_NORMAL。"""
#     global _PATCHED
#     if _PATCHED:
#         return

#     import bayesnf.inference as inference
#     import bayesnf.models as models
#     import bayesnf.spatiotemporal as st

#     # 0) 讓 bayesnf 內部所有 LikelihoodDist(...) 的檢查接受 MIXTURE_NORMAL
#     models.LikelihoodDist = LikelihoodDistPatched
#     # 有些版本 inference 會把 LikelihoodDist import 到自己模組命名空間
#     inference.LikelihoodDist = LikelihoodDistPatched

#     # 1) patch BayesianNeuralFieldEstimator._model_args：混和模型多帶 output_dim=2
#     _orig_model_args = st.BayesianNeuralFieldEstimator._model_args

#     def _model_args_patched(self, batch_shape):
#         args = _orig_model_args(self, batch_shape)
#         if getattr(self, "observation_model", None) == "MIXTURE_NORMAL":
#             args["output_dim"] = 2
#         return args

#     st.BayesianNeuralFieldEstimator._model_args = _model_args_patched

#     # 2) patch inference.make_model：看到 output_dim=2 就用 2-head MLP
#     _orig_make_model = inference.make_model

#     def make_model_patched(*, output_dim=1, **kwargs):
#         if int(output_dim) != 2:
#             # 移除 output_dim，避免原本 make_model 不吃這個參數
#             kwargs.pop("output_dim", None)
#             return _orig_make_model(**kwargs)

#         init_x = kwargs["init_x"]
#         mlp = BayesianNeuralField1D_Mixture(
#             width=kwargs["width"],
#             depth=kwargs["depth"],
#             input_scales=kwargs["input_scales"],
#             fourier_degrees=kwargs["fourier_degrees"],
#             interactions=kwargs["interactions"],
#             num_seasonal_harmonics=kwargs["num_seasonal_harmonics"],
#             seasonality_periods=kwargs["seasonality_periods"],
#         )
#         x0 = jnp.zeros(init_x, dtype=jnp.float32)
#         mlp_template = mlp.init(jax.random.PRNGKey(0), x0)
#         return mlp, mlp_template

#     inference.make_model = make_model_patched

#     # 3) patch models.make_likelihood_model：distribution=MIXTURE_NORMAL 走我們的 mixture
#     _orig_make_like = models.make_likelihood_model

#     def make_likelihood_patched(params, x, mlp, mlp_template, distribution):
#         # distribution 可能是 "MIXTURE_NORMAL" 或 LikelihoodDistPatched.MIXTURE_NORMAL
#         dist = getattr(distribution, "value", distribution)

#         if dist == "MIXTURE_NORMAL":
#             return _make_likelihood_mixture(params, x, mlp, mlp_template)

#         # 其他分配：丟回原本，並用字串 dist 避免 enum 型別造成原碼判斷失敗
#         return _orig_make_like(params, x, mlp, mlp_template, dist)

#     models.make_likelihood_model = make_likelihood_patched

#     _PATCHED = True

def enable_mixture_normal():
    """執行一次即可，會在 runtime 內把 bayesnf 的 hook 接上 (膨脹點 + 高斯) 模型。"""
    global _PATCHED
    if _PATCHED:
        return

    import bayesnf.inference as inference
    import bayesnf.models as models
    import bayesnf.spatiotemporal as st

    # 0) 讓 bayesnf 內部 LikelihoodDist(...) 的檢查接受 MIXTURE_NORMAL / INFLATED_NORMAL
    models.LikelihoodDist = LikelihoodDistPatched
    inference.LikelihoodDist = LikelihoodDistPatched

    # 1) patch BayesianNeuralFieldEstimator._model_args：此模型需要 output_dim=2 (F 與 logit_p)
    _orig_model_args = st.BayesianNeuralFieldEstimator._model_args

    def _model_args_patched(self, batch_shape):
        args = _orig_model_args(self, batch_shape)
        if getattr(self, "observation_model", None) in ("MIXTURE_NORMAL", "INFLATED_NORMAL"):
            args["output_dim"] = 2
        return args

    st.BayesianNeuralFieldEstimator._model_args = _model_args_patched

    # 2) patch inference.make_model：output_dim=2 用 2-head MLP
    _orig_make_model = inference.make_model

    def make_model_patched(*, output_dim=1, **kwargs):
        if int(output_dim) != 2:
            kwargs.pop("output_dim", None)
            return _orig_make_model(**kwargs)

        init_x = kwargs["init_x"]
        mlp = BayesianNeuralField1D_Mixture(
            width=kwargs["width"],
            depth=kwargs["depth"],
            input_scales=kwargs["input_scales"],
            fourier_degrees=kwargs["fourier_degrees"],
            interactions=kwargs["interactions"],
            num_seasonal_harmonics=kwargs["num_seasonal_harmonics"],
            seasonality_periods=kwargs["seasonality_periods"],
        )
        x0 = jnp.zeros(init_x, dtype=jnp.float32)
        mlp_template = mlp.init(jax.random.PRNGKey(0), x0)
        return mlp, mlp_template

    inference.make_model = make_model_patched

    # 3) patch models.make_likelihood_model：兩種名字都導到「膨脹點 + 高斯」
    _orig_make_like = models.make_likelihood_model

    def make_likelihood_patched(params, x, mlp, mlp_template, distribution):
        dist = getattr(distribution, "value", distribution)

        if dist in ("MIXTURE_NORMAL", "INFLATED_NORMAL"):
            return _make_likelihood_inflated_normal(params, x, mlp, mlp_template)

        return _orig_make_like(params, x, mlp, mlp_template, dist)

    models.make_likelihood_model = make_likelihood_patched

    _PATCHED = True

def predict_F_p(estimator, table):
    """從已 fit 的 estimator 取出 F(s,t) 與 p(s,t) 的 posterior samples 與平均。
    回傳：
      F_samples: (n_draws, n_obs)
      p_samples: (n_draws, n_obs)
      F_mean: (n_obs,)
      p_mean: (n_obs,)
    """
    # if getattr(estimator, "observation_model", None) != "MIXTURE_NORMAL":
    #     raise ValueError("estimator.observation_model 不是 MIXTURE_NORMAL")
    obs = getattr(estimator, "observation_model", None)
    if obs not in ("MIXTURE_NORMAL", "INFLATED_NORMAL"):
        raise ValueError(f"estimator.observation_model 需為 MIXTURE_NORMAL 或 INFLATED_NORMAL，目前是 {obs}")

    import bayesnf.inference as inference

    test_data = estimator.data_handler.get_test(table)
    mlp, mlp_template = inference.make_model(**estimator._model_args(test_data.shape))

    # 跟原本 likelihood_model 一樣的 apply mapping
    for _ in range(estimator._ensemble_dims - 1):
        mlp.apply = jax.vmap(mlp.apply, in_axes=(0, None))
    mlp.apply = jax.pmap(mlp.apply, in_axes=(0, None))

    params = estimator.params_
    treedef = jax.tree_util.tree_structure(mlp_template)
    mlp_params = jax.tree_util.tree_unflatten(treedef, params[3:])

    preds = mlp.apply(mlp_params, jnp.array(test_data))  # (..., n_obs, 2)
    F = preds[..., 0]
    logit_p = preds[..., 1]
    mixing_bias = params[2][..., jnp.newaxis]
    p = jax.nn.sigmoid(logit_p + mixing_bias)

    F_np = np.array(F)
    p_np = np.array(p)

    # 把所有 ensemble 維度攤平，只留最後一軸是 observation
    F_samples = F_np.reshape((-1, F_np.shape[-1]))
    p_samples = p_np.reshape((-1, p_np.shape[-1]))

    F_mean = F_samples.mean(axis=0)
    p_mean = p_samples.mean(axis=0)
    return F_samples, p_samples, F_mean, p_mean
