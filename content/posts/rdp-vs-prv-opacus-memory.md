---
author:
  name: "apsal"
date: 2026-06-05
linktitle: rdp-vs-prv-opacus-memory
type:
- post
- posts
title: "Privacy accounting in DP-SGD"
weight: 10
series:
---

I was reproducing the Likelihood Ratio Attack (LiRA) from Carlini et al.'s [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). It was an experiment I had started a long back and Eleuther AI's [Summer of Open AI Research (SOAR)](https://www.eleuther.ai/soar) insipired me to wrap it up before the application deadline. LiRA is a membership inference attack: given a trained model and a data point, can you tell whether that point was in the training set? The attack works by training many shadow models (copies of the target model on different random subsets of the data) and using how the per-sample loss varies across those copies to answer the question. To reproduce the paper's DP-SGD experiments, I ran a grid of shadow models at different noise levels using Opacus on a WSL2 VM with 7.4 GB RAM.

The training runs kept dying around epoch 20. After some debugging I pinned it to a single `privacy_engine.get_epsilon()` call and specifically to which privacy accountant Opacus was using under the hood. That sent me into understanding how privacy accountants actually work. Here's what I found.

## What differential privacy means

Differential privacy is a mathematical guarantee that your model's behavior doesn't change noticeably based on whether any single training example was included.

Two datasets are called *adjacent* if they differ by exactly one example — one has Alice's data, the other doesn't. An algorithm M is (ε, δ)-differentially private if, for any two adjacent datasets D and D' and any set of possible outputs S:

$$P[M(D) \in S] \leq e^\varepsilon \cdot P[M(D') \in S] + \delta$$

ε is the privacy budget. Smaller ε means the output distributions on adjacent datasets are closer together, which means an attacker watching the model has less power to infer who was in the training set. δ is a small probability that the guarantee fails entirely — in practice something like 10⁻⁵, a technical escape hatch the definition needs rather than a real risk in well-tuned settings.

The guarantee is about the worst case: over all possible outputs, over all possible adjacent datasets. It doesn't promise any individual's data is safe. It says the output distributions are close enough that statistical tests have limited power, regardless of who the attacker targets.

## DP-SGD

DP-SGD (Abadi et al., 2016) is the standard way to train neural networks with a differential privacy guarantee. Normal SGD averages gradients over a batch and applies the update. DP-SGD changes two things.

Per-sample gradient clipping. Instead of averaging raw gradients, the gradient is computed separately for each training example, clipped to a maximum L2 norm C, then averaged. This bounds how much any single example can move the update.

Gaussian noise. After clipping, Gaussian noise calibrated to C and a noise multiplier σ is added to the averaged gradient before the weight update:

$$\tilde{g} = \frac{1}{B} \left[ \sum_i \text{clip}(g_i, C) \right] + \mathcal{N}\!\left(0,\; \frac{\sigma^2 C^2}{B^2} I\right)$$

Higher σ means more noise, stronger privacy, and a worse model. C controls sensitivity — how much a single gradient can shift the update.

Opacus also uses Poisson subsampling instead of fixed-size batches. Each training step includes each example independently with probability q = batch_size / dataset_size, rather than sampling a fixed batch. Poisson subsampling amplifies privacy (a subsampled mechanism is more private than the same mechanism on the full batch), and this gets built into the accounting.

Now the accounting question: after T optimizer steps with noise σ and clipping norm C, what is the actual (ε, δ)?

## The composition problem

Each optimizer step is one application of the Gaussian mechanism. T steps compose T times. The naive bound, basic composition, says if each step costs ε₀ then T steps cost T·ε₀. That bound is correct but useless in practice. At any practical noise level, T·ε₀ blows past a useful privacy budget within a few epochs.

The better approach is to exploit the specific shape of the Gaussian mechanism's privacy-loss distribution. Rather than bounding the worst case at each step independently, you track that distribution across compositions. RDP and PRV both do this, but through different methods with very different memory costs.

## RDP accounting

Rényi DP (Mironov 2017) reframes the privacy guarantee using Rényi divergence. The Rényi divergence of order α between distributions P and Q measures how different they are, parameterized by how much weight you put on the tails. At α = 1 it recovers KL divergence; higher α emphasizes the heavy tail behavior. Formally:

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log \mathbb{E}_Q\!\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

A mechanism M is (α, ε)-RDP if D_α(M(D) || M(D')) ≤ ε for all adjacent D, D'. Composition in Rényi terms is clean: T applications of an (α, ε)-RDP mechanism give (α, T·ε)-RDP. Still linear, but the per-step ε in Rényi terms is much smaller than in (ε, δ) terms for the Gaussian mechanism at practical noise levels, so the budget lasts.

To convert back to standard (ε, δ)-DP at the end:

$$\varepsilon(\delta) = \min_\alpha \left[ \mathrm{RDP}_\alpha - \frac{\log \delta}{\alpha - 1} \right]$$

The minimum over α matters because different orders are optimal at different step counts and δ targets. Opacus evaluates at 151 Rényi orders and takes the best.

Inside `RDPAccountant.get_privacy_spent`, `compute_rdp` is called once per segment in history, results are summed, then converted to (ε, δ)-DP:

```python
# opacus/accountants/rdp.py
rdp = sum(
    compute_rdp(q=sample_rate, noise_multiplier=noise_multiplier,
                steps=num_steps, orders=alphas)
    for (noise_multiplier, sample_rate, num_steps) in self.history
)
eps, best_alpha = get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
```

The memory behavior becomes clear inside `compute_rdp`. It calls `_compute_rdp` once per Rényi order to get the per-step RDP value, then scales by `steps`:

```python
# opacus/accountants/analysis/rdp.py

def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    if q == 1.0:
        return alpha / (2 * sigma**2)      # no subsampling: plain Gaussian
    return _compute_log_a(q, sigma, alpha) / (alpha - 1)

def compute_rdp(q, noise_multiplier, steps, orders):
    rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])
    return rdp * steps  # steps is a scalar — output is always shape (151,)
```

`steps` never enters an array dimension. It scales the fixed-size output after the per-step computation is done. That's why memory doesn't grow with training length.

The per-step RDP at order α reduces to `_compute_log_a(q, σ, α) / (α - 1)`. For integer α, `_compute_log_a` evaluates the binomial expansion from Mironov et al. 2019 §3.3 in log-space:

```python
def _compute_log_a_for_int_alpha(q, sigma, alpha):
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )
        s = log_coef_i + (i * i - i) / (2 * sigma**2)
        log_a = _log_add(log_a, s)  # numerically stable log-space add
    return log_a
```

The loop runs `alpha + 1` times — bounded by the Rényi order, not the step count. Calling this every epoch adds no incremental cost.

The cost of this approach is accuracy. The RDP-to-(ε, δ) conversion is an upper bound on top of the RDP bound, so the reported ε is somewhat larger than the true value. For the same mechanism and the same T, RDP will always report a looser ε than a tighter accountant would.

## PRV accounting

The Privacy Random Variable accountant (Gopi, Lee, Wutschitz, NeurIPS 2021) works with the privacy-loss distribution directly.

For adjacent datasets D and D', the privacy-loss random variable Z captures how much a single output reveals about which dataset was used:

$$Z = \log \frac{P[M(D) = o]}{P[M(D') = o]}$$

where o is drawn from M(D). The (ε, δ)-DP condition turns out to be equivalent to P[Z > ε] ≤ δ. Computing (ε, δ) from Z is then a tail probability question: find ε such that the probability mass above ε equals δ.

Composing T steps means summing T independent copies of Z (Z₁ + Z₂ + ... + Z_T gives the total privacy loss across T steps). Summing independent random variables means convolving their distributions. Convolution in frequency space is pointwise multiplication. Hence the FFT:

```python
composed_pmf = irfft(rfft(dprv.pmf) ** num_steps)
```

This is more accurate than RDP because it works with the actual composed distribution rather than a divergence bound.

But to do any of this numerically you have to discretize the distribution onto a finite grid, and that's where the memory cost comes in.

### The grid

The grid needs to cover the tails of the composed distribution. Gopi et al. derive the required domain half-width `L_max` from an RDP bound over all T compositions (Remark 5.6 of the paper). Because that RDP bound grows with T, `L_max` grows with T. The grid spacing `dt` is fixed by the requested error tolerances. So:

$$\text{grid\_size} \approx \frac{2 \cdot L_{\max}}{dt}$$

Every call to `get_epsilon` at a larger T builds a bigger grid, runs a bigger FFT, and allocates more working memory. Here's what that looks like with no GPU — just the accounting math:

```python
import gc, psutil
from opacus.accountants import create_accountant

def watch(mechanism, steps_per_epoch=98, epochs=12):
    acc = create_accountant(mechanism=mechanism)
    acc.history = [(0.2, 256/25000, 0)]
    rss = lambda: psutil.Process().memory_info().rss / 1024**2
    for ep in range(1, epochs + 1):
        nm, sr, _ = acc.history[0]
        acc.history[0] = (nm, sr, ep * steps_per_epoch)
        acc.get_epsilon(1e-5)
        print(f"{mechanism} epoch {ep:2d}: steps={ep*steps_per_epoch:5d}  RSS={rss():.0f} MB")
    gc.collect()

watch("prv")
watch("rdp")
```

```
prv epoch  1: steps=196    RSS=552 MB
prv epoch  2: steps=392    RSS=755 MB
prv epoch  3: steps=588    RSS=1033 MB
prv epoch  4: steps=784    RSS=1685 MB
prv epoch  5: steps=980    RSS=2487 MB
prv epoch  6: steps=1176   RSS=3016 MB

rdp epoch  1: steps=196    RSS=401 MB
rdp epoch  6: steps=1176   RSS=402 MB
```

The `gc.collect()` after the PRV loop reclaims nothing. `scipy.fft` caches FFT plans keyed by transform size, and each new grid size adds a plan at the C level, below Python's GC. Glibc also typically doesn't return freed heap arenas to the OS when each successive allocation is larger than the previous one, since the old block can't be reused. The resident set ratchets up and stays.

## Tradeoffs

PRV gives a tighter ε. For the same mechanism and the same T, it reports a smaller ε than RDP. On long training runs this difference can be meaningful — PRV might report ε = 8 where RDP reports ε = 11. That accuracy comes from working with the actual privacy-loss distribution rather than a Rényi surrogate.

RDP stays at a fixed (151,) array throughout training. PRV's working memory grows with the step count, and called every epoch on a long run it can reach gigabytes. That's what killed my training runs.

For most experimental work, RDP is the right choice: the bound is correct, the memory is flat, and it's what the LiRA and TF-Privacy reference implementations use, so the numbers are directly comparable. If you're reporting ε in a paper where the tightness matters — comparing against PRV-reported numbers from another paper, or making a compliance argument — use PRV, but call it once at the end of training rather than every epoch:

```python
for epoch in range(num_epochs):
    train_one_epoch(...)

epsilon = privacy_engine.get_epsilon(delta)  # one call, at the end
```

Worth knowing: that final-epoch PRV call is the most expensive one in the run, since the grid is at its largest. On a tight machine, using RDP throughout and doing one optional PRV call at the very end is the safest combination.

One last thing. Switching accountants does not change training at all. The noise multiplier, clipping norm, and gradient updates are identical between accountants. The accountant is purely a post-hoc measurement of the privacy cost. But ε values from RDP and PRV aren't on the same scale, so don't mix them in the same logs.

## References

- M. Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016.
- I. Mironov, *Rényi Differential Privacy*, CSF 2017.
- I. Mironov, K. Talwar, L. Zhang, *Rényi Differential Privacy of the Sampled Gaussian Mechanism*, 2019.
- S. Gopi, Y. T. Lee, L. Wutschitz, *Numerical Composition of Differential Privacy*, NeurIPS 2021.
- A. Yousefpour et al., *Opacus: User-Friendly Differential Privacy Library in PyTorch*, 2021.
- N. Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022.

The LiRA reproduction this came out of is at [github.com/apzl/LiRA-pytorch](https://github.com/apzl/LiRA-pytorch), including shadow model training, DP-SGD grid, scoring pipeline, and a longer debugging writeup.
