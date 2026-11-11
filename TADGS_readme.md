# Understanding TAD-GS: From 3DGS to Temporally Aware Densification

> [!NOTE]
> **Who this is for**: you know **3D Gaussian Splatting (3DGS)**: a scene is a cloud of 3D Gaussians, each with a position, shape, opacity and colour; they are projected to the screen and alpha-blended; and the cloud grows during training by cloning and splitting Gaussians. You do **not** need to know any dynamic-scene (4D) methods. Section 1 covers the 3DGS pieces this document relies on, and Section 2 introduces dynamic Gaussian Splatting from scratch.
>
> All code links point to [`TAD-GS/scene/tad_gaussian.py`](TAD-GS/scene/tad_gaussian.py) unless stated otherwise.

---

## 0. Notation Cheat-Sheet

Look things up here as you read.

| Symbol | Meaning | Code name |
| :--- | :--- | :--- |
| $i$ | Index of one Gaussian. There are $N$ Gaussians in total. | row of every `(N, …)` tensor |
| $T$ | Number of frames in the video (e.g. $50$ for `coffee_martini`). | `duration` |
| $t \in [0, 1]$ | **Normalised time**. Frame $k$ corresponds to $t = k / T$. | `viewpoint_camera.timestamp` |
| $k$ | Training **iteration** (one optimiser step). Each iteration renders one camera at one timestamp $t_k$. | `iteration` |
| $\boldsymbol{\mu}_{i,0} \in \mathbb{R}^3$ | Base 3D position (centre) of Gaussian $i$. | `_xyz` |
| $\mathbf{s}_i \in \mathbb{R}^3$ | 3D scale (size along its 3 axes); stored as a log, so the real scale is $e^{\text{stored value}}$. | `_scaling` |
| $\mathbf{q}_{i,0} \in \mathbb{R}^4$ | Base rotation as a **quaternion**, i.e. 4 numbers $(w,x,y,z)$ that encode a 3D rotation once scaled to unit length. | `_rotation` |
| $\tilde{o}_i \in \mathbb{R}$ | Raw (pre-activation) opacity. A tilde $\tilde{\ }$ always means "raw value before an activation function". | `_opacity` |
| $\mathrm{sigm}(x) = \frac{1}{1+e^{-x}}$ | Sigmoid; squashes any number into $(0,1)$. | `torch.sigmoid` |
| $\mathbf{c}_i$ | Colour as **spherical-harmonic (SH)** coefficients (view-dependent colour), exactly as in 3DGS. | `_features_dc`, `_features_rest` |
| $\mu_{t,i} \in [0,1]$ | **Temporal centre**: the moment in time Gaussian $i$ is "about". Fixed, not learned. | `_trbf_center` |
| $\delta_i(t)$ | **Time offset** of $t$ from $\mu_{t,i}$ (possibly warped, see TOW). | `get_offset()` |
| $`\sigma_{i,L}, \sigma_{i,R}`$ | **Temporal widths**: how long Gaussian $i$ takes to fade in (Left, before $\mu_{t,i}$) and fade out (Right, after). | `get_trbf_scale` |
| $o_i(t) \in [0,1]$ | Opacity of Gaussian $i$ at time $t$. | `curr_opacity` |
| $`\boldsymbol{\mu}_i(t),\ \mathbf{q}_i(t)`$ | Position and rotation of Gaussian $i$ at time $t$. | `get_xyz_at_t()`, `get_rotation_at_t()` |
| $\mathbf{p}_{i}$ | 2D screen-space (pixel) position of Gaussian $i$'s projected centre. | `viewspace_point_tensor` |
| $`\nabla_{\mathbf{p}_i}\mathcal{L}`$ | Gradient of the image loss $\mathcal{L}$ w.r.t. $\mathbf{p}_i$: "how much the loss wants this Gaussian to move on screen". | `viewspace_point_tensor.grad[:, :2]` |
| $\lVert\cdot\rVert_2$ | Euclidean length of a vector. | `torch.norm` |
| $\tau_{\text{pos}}$ | Densification gradient threshold (default $0.0002$; $0.0003$ for `coffee_martini`). | `densify_grad_threshold` |
| $\mathrm{clamp}(x, a, b)$ | Clip $x$ into the interval $[a,b]$. | `torch.clamp` |
| $\mathrm{sign}(x)$ | $+1$ if $`x>0`$, $-1$ if $`x<0`$, $0$ if $x=0$. | `torch.sign` |

> [!IMPORTANT]
> The original 3DGS papers write opacity as $\alpha$. **This document writes opacity as $o$** because $\alpha$ is also the name of a TAT hyperparameter, which is renamed to $\gamma$ here. In the same way, $\sigma$ (no argument) always means a **temporal width**, never the sigmoid, which is written $\mathrm{sigm}$.

---

## 1. Refresher: The Parts of 3DGS That Matter Here

### 1.1 A static 3D Gaussian

In 3DGS, Gaussian $i$ has the parameters

```math
\Theta_i^{\text{3DGS}} = \big\{\, \boldsymbol{\mu}_{i,0},\ \mathbf{s}_i,\ \mathbf{q}_{i,0},\ \tilde{o}_i,\ \mathbf{c}_i \,\big\}
```

Its shape is the covariance $`\boldsymbol{\Sigma}_i = R(\mathbf{q})\,S\,S^\top R(\mathbf{q})^\top`$, built from the rotation matrix $R(\mathbf{q})$ and the diagonal scale matrix $S = \mathrm{diag}(\mathbf{s})$. To render, every Gaussian is projected to a 2D ellipse on screen, the ellipses are sorted front to back, and colours are alpha-blended using opacity $`o_i = \mathrm{sigm}(\tilde{o}_i)`$.

The rasterizer also returns, for each Gaussian, its on-screen radius $\text{radii}_i$. It is $`>0`$ whenever the Gaussian lands inside the camera's view frustum (the pyramid-shaped region the camera can see). In code this is `visibility_filter = radii > 0` ([renderer](TAD-GS/gaussian_renderer/__init__.py#L94)).

### 1.2 Adaptive Density Control (clone / split)

Training starts from a sparse point cloud. To add detail, 3DGS periodically adds Gaussians where the reconstruction is poor:

1. **Accumulate a signal.** After every iteration, for every visible Gaussian, add $`\lVert\nabla_{\mathbf{p}_i}\mathcal{L}\rVert_2`$ to a running sum and add $1$ to a counter. A large screen-space gradient means the loss keeps pulling this Gaussian around: it is trying to cover a region it cannot represent on its own.
2. **Decide.** Every `densification_interval` iterations (200 in this repo), compute the average gradient $G_i = \text{sum}/\text{counter}$. If $`G_i \ge \tau_{\text{pos}}`$:
   - **Clone** it (make an identical copy) if it is *small* (max scale $\le$ `percent_dense` $\times$ scene extent). This handles under-reconstruction.
   - **Split** it into 2 smaller Gaussians (scale divided by $1.6$, positions sampled inside the parent) if it is *large*. This handles over-reconstruction.
3. **Prune.** Remove Gaussians whose opacity has dropped below a threshold.

Densification is what decides **where detail gets added**, so if this signal is wrong, some regions never get enough Gaussians. This is the heart of TAD-GS.

---

## 2. From 3DGS to Dynamic Scenes (4DGS in One Page)

For a **video** (multi-view, with many frames over time), each Gaussian has to change over time. There are two families of methods:

1. **Deformation-field methods** (e.g. 4D-GS by Wu et al., Deformable-3DGS):
   - Keep one *canonical* (reference) set of 3D Gaussians.
   - For each time $t$, a small neural network (often reading from a learned feature grid such as *HexPlane*, a set of 2D feature planes over $`(x,y,z,t)`$) predicts how much each Gaussian should move, rotate and rescale.
   - *Downside*: every frame needs a network pass over all Gaussians, which is slow. Things that suddenly **appear or disappear** (pouring liquid, flames) are also hard to model, because every Gaussian exists at every time and can only be moved around.
2. **Explicit per-Gaussian methods** (e.g. Spacetime Gaussians, STG):
   - No network. Each Gaussian stores its own small closed-form functions of time: a **trajectory** $\boldsymbol{\mu}_i(t)$, a **rotation** $\mathbf{q}_i(t)$, and a **temporal opacity** $o_i(t)$ that rises and falls like a bell curve, so a Gaussian can be "alive" only for part of the video.
   - *Upside*: very fast, and appearing/disappearing objects are natural (opacity just goes to 0).

**TAD-GS builds on the explicit family.** The rest of this document uses one key consequence of that choice: **each Gaussian is only "alive" (visible) for part of the video.**

---

## 3. The Core Insight of TAD-GS

Prior dynamic methods redesigned how Gaussians **move**, but **kept 3DGS's densification rule unchanged**. That rule assumes something that is only true for static scenes:

- **In 3DGS**, a Gaussian inside the camera frustum is genuinely visible, so averaging its gradient over all frustum-visible iterations makes sense.
- **In a video**, a Gaussian can be inside the frustum while being **temporally invisible**. A Gaussian modelling a hand at frame 24 may have opacity $o_i \approx 0.001$ at frame 4, but its 3D position still projects inside the image at frame 4, so 3DGS counts frame 4 as a "visible" frame.
- **Lifespans vary a lot.** Static background Gaussians live for the whole video (e.g. all 50 frames); Gaussians for fast motion may live for only a few frames.

The result is that **short-lived, fast-moving Gaussians have their gradient averaged over many frames where they don't actually exist**. Their average stays below $\tau_{\text{pos}}$, so they never get densified, and fast motion stays blurry. Static background, meanwhile, gets densified normally.

**TAD-GS** (*Temporally Aware Densification for Dynamic 3D Gaussian Splatting*, ECCV 2026) fixes this with two densification changes (Part B). The codebase also includes a per-Gaussian temporal representation (Part A), which these changes rely on.

```text
 PART A: per-Gaussian temporal representation        (Section 4)
 ┌─────────────────────────────────────────────────────────────────┐
 │                  Temporal Offset Warping (TOW)                  │
 │                  time t  ──►  warped offset δ_i(t)              │
 │                     │                         │                 │
 │                     ▼                         ▼                 │
 │   Asymmetric temporal opacity       Fourier motion  μ_i(t)      │
 │   o_i(t), widths σ_L, σ_R           + linear rotation q_i(t)    │
 └─────────────────────────────────────────────────────────────────┘
                                │
                                │  render at time t, backpropagate,
                                │  collect o_i(t), σ_i(t), 2D gradients
                                ▼
 PART B: temporally aware densification              (Section 6)
 ┌─────────────────────────────────────────────────────────────────┐
 │   Visibility-Aware Densification (VAD)                          │
 │   opacity-weighted gradient average, ghost frames dropped       │
 │                     │                                           │
 │                     ▼                                           │
 │   Temporally Adaptive Thresholding (TAT)                        │
 │   lower clone/split threshold for short-lived Gaussians         │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 4. Part A: How One Gaussian Changes Over Time

A TAD-GS Gaussian ([`GaussianModel`](TAD-GS/scene/tad_gaussian.py#L27)) keeps the 3DGS parameters and adds temporal ones:

| Parameter | Shape per Gaussian | Changes with time? |
| :--- | :--- | :--- |
| Scale $\mathbf{s}_i$, colour $\mathbf{c}_i$ | same as 3DGS | **No**, constant |
| Temporal centre $\mu_{t,i}$ | 1 | No (fixed, not optimised) |
| Temporal widths $`(\tilde{s}_{i,L}, \tilde{s}_{i,R})`$ | 2 | Used to compute $o_i(t)$ |
| Motion coefficients $`\mathbf{a}_{i,f}, \mathbf{b}_{i,f}`$ | $4 \times 2 \times 3 = 24$ | Used to compute $\boldsymbol{\mu}_i(t)$ |
| Phase shifts $\phi_{i,f}$ | 4 | Used to compute $\boldsymbol{\mu}_i(t)$ |
| Rotation velocity $\boldsymbol{\omega}_i$ | 4 | Used to compute $\mathbf{q}_i(t)$ |

**Where does $\mu_{t,i}$ come from?** The initial point cloud is built from a few *anchor frames* (every 24th frame by default: 0, 24, 48). Each point's $\mu_{t,i}$ is the normalised time of the frame it came from ([init](TAD-GS/scene/tad_gaussian.py#L273)). Cloned/split children inherit it from their parent. It is never optimised.

At render time $t$, the renderer asks each Gaussian for its position, rotation and opacity *at that time* ([renderer](TAD-GS/gaussian_renderer/__init__.py#L67-L69)) and then rasterises exactly like 3DGS. All three quantities depend on a single number: **the time offset $\delta_i(t)$**.

---

### 4.1 Temporal Offset Warping (TOW, `--tow`): the time offset $\delta_i(t)$

**Without TOW**, the offset is just how far $t$ is from the Gaussian's centre:

```math
\delta_i(t) = t - \mu_{t,i}
```

**The problem.** Take $T = 50$ frames, with a Gaussian that only matters within a window of $S = 10$ frames around $\mu_{t,i}$. In normalised time that window is $10/50 = 0.2$ wide. The motion model (Section 4.3) is a sum of sines and cosines of $\delta$ with periods $1, \tfrac12, \tfrac13, \tfrac14$. Over a window only $0.2$ wide, even the fastest harmonic (period $\tfrac14$) completes less than one cycle, so **the Gaussian cannot express fast back-and-forth motion inside the window where it actually lives**. Most of each wave's shape is "spent" on times when the Gaussian is invisible anyway.

**The fix.** Stretch time inside the window and compress it outside, so that most of the offset range falls inside the window. Two hyperparameters control this:

- $S$ = `temporal_span` (default $10$ frames): window width. In normalised units, its half-width is $m = \dfrac{S}{2T}$ (here $m = 0.1$).
- $B$ = `span_budget` (default $70$): the **percentage** of the total offset range given to the window.

These define two slopes ("how fast warped time runs"):

```math
\lambda_{\text{in}} = \frac{B}{100}\cdot\frac{T}{S}, \qquad \lambda_{\text{out}} = \frac{100 - B}{100}\cdot\frac{T}{T - S}
```

The warped offset is a **piecewise-linear** function (two straight-line pieces joined at $\pm m$), implemented in [`offset_warping_vectorized`](TAD-GS/scene/tad_gaussian.py#L804-L863) and called from [`get_offset`](TAD-GS/scene/tad_gaussian.py#L111-L125). Writing $d = t - \mu_{t,i}$ for the plain offset:

```math
\delta_i(t) = \begin{cases}
d \cdot \lambda_{\text{in}}, & |d| \le m \quad \text{(inside the window: steep slope)}\\[6pt]
\mathrm{sign}(d)\, m\, \lambda_{\text{in}} \;+\; \big(d - \mathrm{sign}(d)\, m\big)\, \lambda_{\text{out}}, & |d| > m \quad \text{(outside: shallow slope)}
\end{cases}
```

The second line reads: "go to the window edge at the steep slope, then continue the rest of the way at the shallow slope". Both pieces meet at $d = \pm m$, so $\delta$ is continuous.

> [!TIP]
> **Worked numbers** ($T=50$, $S=10$, $B=70$): $\lambda_{\text{in}} = 0.7 \cdot 5 = 3.5$ and $\lambda_{\text{out}} = 0.3 \cdot 50/40 = 0.375$.
> - The window ($0.2$ of normalised time) maps to an offset range of $0.2 \times 3.5 = 0.7$.
> - The remaining $0.8$ of time maps to $0.8 \times 0.375 = 0.3$.
>
> The total range is still $1$, but **70% of it now sits inside the 10 frames where the Gaussian lives**. The fastest harmonic now completes about $0.7 \times 4 = 2.8$ cycles inside the window instead of $0.8$. Sharper motion, with **no extra learnable parameters**.

*Edge case:* if a Gaussian's centre is exactly at the start ($\mu_{t,i}=0$) or end of the video, the code slides the window inward so it stays inside $[0,1]$.

---

### 4.2 Asymmetric Temporal Opacity: $o_i(t)$

**Idea.** A Gaussian's opacity should rise and fall over time like a bell curve centred at $\mu_{t,i}$. Real events are often lopsided: a splash appears suddenly but fades slowly. So TAD-GS uses **different widths on each side** of the peak ([`deform_gauss_op_asym`](TAD-GS/scene/tad_gaussian.py#L188-L204), [`get_opacity_at_t`](TAD-GS/scene/tad_gaussian.py#L214-L218)):

```math
o_i(t) = \underbrace{\mathrm{sigm}(\tilde{o}_i)}_{\text{peak opacity (as in 3DGS)}} \cdot \underbrace{\exp\!\left( -\left( \frac{\delta_i(t)}{\sigma_i(t)} \right)^{2} \right)}_{\text{temporal bell, between 0 and 1}},
\qquad
\sigma_i(t) = \begin{cases} \sigma_{i,L} & \text{if } \delta_i(t) < 0 \ \ \text{(before the peak)}\\ \sigma_{i,R} & \text{if } \delta_i(t) \ge 0 \ \ \text{(after the peak)} \end{cases}
```

- At $\delta = 0$ (the Gaussian's own moment) the bell equals $1$, so $`o_i = \mathrm{sigm}(\tilde{o}_i)`$.
- Large $\sigma$ means a wide bell: the Gaussian lives a long time (e.g. static background).
- Small $\sigma$ means a narrow bell: a short-lived, transient Gaussian.

The $\sigma$ that was used for the current frame is saved (`current_trbf_scale`) because TAT needs it later.

#### Keeping $\sigma$ from collapsing: the $\sigma_{\min}$ scale

**Problem.** If $\sigma_i$ becomes tiny, the Gaussian is visible in exactly one training frame. It memorises that frame and vanishes at the in-between times you want to interpolate. This is a "Dirac spike": a function that is non-zero at only one instant.

**Fix.** Express every $\sigma$ in units of a minimum width $\sigma_{\min}$, chosen so that a Gaussian of width $\sigma_{\min}$ has its temporal bell above a visibility level $\tau_{\text{vis}}$ for at least $M$ consecutive frames ([`sigma_min_for_M_frames`](TAD-GS/scene/tad_gaussian.py#L773-L784)).

*Derivation.* The bell is $\ge \tau_{\text{vis}}$ when $|\delta| \le \sigma\sqrt{-\ln \tau_{\text{vis}}}$. Frames are spaced $\frac{1}{T-1}$ apart, so $M$ frames span a half-width of $h = \frac{M-1}{2(T-1)}$. Setting $\sigma\sqrt{-\ln\tau_{\text{vis}}} = h$ gives

```math
\sigma_{\min}(M, T, \tau_{\text{vis}}) = \frac{M - 1}{2(T - 1)\sqrt{-\ln \tau_{\text{vis}}}}
```

With the defaults $M = 10$ (`min_visible_timestamps`), $\tau_{\text{vis}} = 0.2$ (`visibility_threshold`) and $T = 50$, this gives $\sigma_{\min} \approx 0.072$.

The stored raw parameters $`\tilde{s}_{i,L}, \tilde{s}_{i,R}`$ (`_trbf_scale`) are turned into widths by ([`get_trbf_scale`](TAD-GS/scene/tad_gaussian.py#L182-L186)):

```math
\sigma_{i,L} = \mathrm{clamp}\!\big( \sigma_{\min} \cdot e^{\tilde{s}_{i,L}},\ 0,\ 1.1 \big), \qquad \sigma_{i,R} = \mathrm{clamp}\!\big( \sigma_{\min} \cdot e^{\tilde{s}_{i,R}},\ 0,\ 1.1 \big)
```

> [!WARNING]
> **$\sigma_{\min}$ is a soft scale, not a hard floor.** The raw values start at $\tilde{s} = 1$ (`trbfslinit`), so every Gaussian starts with $\sigma = e \cdot \sigma_{\min} \approx 0.20$. But $\tilde{s}$ is unconstrained: if the optimiser pushes it negative, $`e^{\tilde{s}} < 1`$ and $\sigma$ falls below $\sigma_{\min}$, and it can approach $0$ (never reach it). What the reparameterisation gives you is a *sensible starting width and step size*, measured in "minimum-visible-window" units.
>
> Also, $\sigma_{\min}$ is derived for the **un-warped** offset. With TOW on, the offset moves $3.5\times$ faster per frame inside the window, so a Gaussian of width $\sigma_{\min}$ is in practice visible above $\tau_{\text{vis}}$ for fewer than $M$ frames.

---

### 4.3 Fourier Motion and Linear Rotation: $\boldsymbol{\mu}_i(t)$, $\mathbf{q}_i(t)$

**Why not polynomials or a network?** STG moves each Gaussian along a cubic polynomial $`\sum_{k=1}^{3}\mathbf{b}_k\,(t-\mu_t)^k`$. Polynomials grow without bound as you move away from the centre, so a Gaussian can fly off when evaluated outside its training range. A neural network avoids that but is slow. **Sines and cosines are bounded** (always between $-1$ and $+1$), so they cannot blow up, and they are cheap to evaluate.

**Trajectory** ([`deform_fourier_asym`](TAD-GS/scene/tad_gaussian.py#L147-L160), [`get_xyz_at_t`](TAD-GS/scene/tad_gaussian.py#L171-L173)). The position is the base position plus a **Fourier series**: a sum of $F$ waves of increasing frequency (*harmonics*). Here $F = 4$ (`motion_degree`):

```math
\boldsymbol{\mu}_i(t) = \boldsymbol{\mu}_{i,0} + \sum_{f=1}^{F} \Big[\, \mathbf{a}_{i,f} \cos\!\big(2\pi f\,\delta_i(t) + \phi_{i,f}\big) \;+\; \mathbf{b}_{i,f} \sin\!\big(2\pi f\,\delta_i(t) + \phi_{i,f}\big) \Big]
```

- $f = 1,2,3,4$: harmonic number. Harmonic $f$ repeats every $1/f$ units of $\delta$, so higher $f$ means faster wiggles.
- $`\mathbf{a}_{i,f}, \mathbf{b}_{i,f} \in \mathbb{R}^3`$: how far (in $x,y,z$) the cosine and sine waves of harmonic $f$ push the Gaussian (`_motion`, $24$ numbers).
- $\phi_{i,f}$: a **phase shift**, which slides harmonic $f$'s wave left or right in time (`_amplitude`; the name is misleading, since it is a phase, not an amplitude).

Note that the displacement is generally **not zero at $\delta = 0$**, so $\boldsymbol{\mu}_{i,0}$ is a base point, not necessarily where the Gaussian sits at its own centre time.

**Rotation** ([`get_rotation_at_t`](TAD-GS/scene/tad_gaussian.py#L136-L139)). The quaternion moves in a straight line in 4D quaternion space and is then rescaled to unit length (a valid rotation):

```math
\mathbf{q}_i(t) = \frac{\mathbf{q}_{i,0} + \delta_i(t)\,\boldsymbol{\omega}_i}{\big\lVert\mathbf{q}_{i,0} + \delta_i(t)\,\boldsymbol{\omega}_i\big\rVert_2}
```

$\boldsymbol{\omega}_i \in \mathbb{R}^4$ (`_omega`) is a velocity *in quaternion space*. It is often loosely called "angular velocity", but it is not a physical $\mathbb{R}^3$ angular velocity, and because of the normalisation the actual rotation speed is not constant.

---

## 5. Why 3DGS Densification Fails on Videos

Now back to densification, using the notation above. With `--vad` **off**, TAD-GS uses the plain 3DGS rule ([`d_add_densification_stats`, `vad=False`](TAD-GS/scene/tad_gaussian.py#L718-L722)). After each iteration $k$, which renders one camera at time $t_k$, every Gaussian with $`\text{radii}_i > 0`$ is updated:

```math
G_i^{\text{3DGS}} = \frac{\sum_{k \in \mathcal{F}_i} \big\lVert\nabla_{\mathbf{p}_i} \mathcal{L}_k\big\rVert_2}{|\mathcal{F}_i|},
\qquad \mathcal{F}_i = \big\{\, k \;:\; \text{radii}_i(k) > 0 \,\big\}
```

Read this as: $\mathcal{F}_i$ is the **set** of iterations in which Gaussian $i$ was inside the frustum, $|\mathcal{F}_i|$ is how many there were, and $G_i$ is the average gradient over those iterations. Every 200 iterations, Gaussians with $`G_i \ge \tau_{\text{pos}}`$ are cloned or split.

### Failure mode 1: "ghost visibility" dilutes the average

The rasterizer sets $`\text{radii}_i > 0`$ based on **geometry only**. It does not care that the Gaussian's temporal opacity is nearly zero.

*Example.* Gaussian $i$ represents a splash around frame 24: $o_i \approx 0.9$ there but $o_i \approx 0.001$ at frame 2.

- At frame 2 the Gaussian contributes almost nothing to the image, so its gradient is $\approx 0$.
- But it still projects into the image, so the counter $|\mathcal{F}_i|$ goes up by 1 anyway.

Suppose that over 100 iterations it is genuinely visible in 10 (gradient $\approx 0.0004$ each) and a "ghost" in 90 (gradient $\approx 0$):

```math
G_i^{\text{3DGS}} \approx \frac{10 \times 0.0004}{100} = 0.00004 \;\ll\; \tau_{\text{pos}} = 0.0002
```

The Gaussian clearly needs more detail (its real gradient is $2\times$ the threshold) but is **diluted 10×** and never densified.

### Failure mode 2: one threshold for very different lifespans

Even with a correct average, a single threshold $\tau_{\text{pos}}$ treats a long-lived background Gaussian (large $\sigma$) and a short-lived fast-motion Gaussian (small $\sigma$) the same way. The background one collects consistent gradients across the whole video. The short-lived one has only a few frames to "prove" it needs densifying. Fast-moving parts end up under-reconstructed.

---

## 6. Part B: TAD-GS's Two Densification Fixes

Both live in [`d_add_densification_stats`](TAD-GS/scene/tad_gaussian.py#L716-L737) (accumulation, called every iteration from [`control_gaussians_n3dv`](TAD-GS/utils/common/helper_train.py#L100)) and [`densify_gaussians`](TAD-GS/scene/tad_gaussian.py#L694-L714) (decision, every 200 iterations).

### 6.1 Visibility-Aware Densification (VAD, `--vad`) fixes failure mode 1

VAD ([code](TAD-GS/scene/tad_gaussian.py#L724-L737)) changes the average in two ways:

1. **Ignore ghost frames**: only count iteration $k$ if the Gaussian is actually visible in time, $`o_i(t_k) > 0.05`$.
2. **Weight by opacity**: frames where the Gaussian is more opaque count more, in both the sum and the count. This is a **weighted average**.

```math
G_i^{\text{VAD}} = \frac{\sum_{k \in \mathcal{V}_i} o_i(t_k)\, \big\lVert\nabla_{\mathbf{p}_i} \mathcal{L}_k\big\rVert_2}{\sum_{k \in \mathcal{V}_i} o_i(t_k)},
\qquad \mathcal{V}_i = \big\{\, k \;:\; \text{radii}_i(k) > 0 \ \text{ and } \ o_i(t_k) > 0.05 \,\big\}
```

($\text{and}$ means both conditions must hold.)

*Same example as before:* the 90 ghost frames ($o \approx 0.001$) are dropped, and the 10 real frames all have $o \approx 0.9$, so

```math
G_i^{\text{VAD}} \approx \frac{10 \times 0.9 \times 0.0004}{10 \times 0.9} = 0.0004 \;\ge\; \tau_{\text{pos}}
```

and the Gaussian is densified.

### 6.2 Temporally Adaptive Thresholding (TAT, `--tat`) fixes failure mode 2

**Step 1: measure each Gaussian's lifespan.** During the same VAD accumulation, the code also averages the temporal width $`\sigma_i(t_k)`$ (left or right, whichever side of the peak that frame was on) over the valid frames:

```math
\bar{\sigma}_i = \frac{1}{|\mathcal{V}_i|}\sum_{k \in \mathcal{V}_i} \sigma_i(t_k)
\qquad (\text{code: } \texttt{trbf\_scale\_accum / denom\_op})
```

If a Gaussian had no valid frames, $\bar{\sigma}_i$ defaults to $1$ (treated as static).

**Step 2: give each Gaussian its own threshold** ([`densify_gaussians`](TAD-GS/scene/tad_gaussian.py#L698-L703), [`grad_normalizer`](TAD-GS/scene/tad_gaussian.py#L865-L871)):

```math
\tau_i = \tau_{\text{pos}} \cdot \Big( \beta + (1 - \beta)\,\mathrm{clamp}(\bar{\sigma}_i, 0, 1)^{\gamma} \Big)
```

with $\beta = 0.75$ (`beta`) and $\gamma = 0.8$ (`alpha` in code). The bracket is a multiplier between $\beta$ and $1$:

- $\beta$ sets the **largest discount**: the most a Gaussian's threshold can be lowered.
- $\gamma$ sets the **curve shape**: how quickly the discount fades as $\bar{\sigma}$ grows.

| Gaussian type | $\bar{\sigma}_i$ | Multiplier | Threshold $\tau_i$ |
| :--- | :--- | :--- | :--- |
| Static / long-lived | $1.0$ | $0.75 + 0.25 \cdot 1 = 1.00$ | $\tau_{\text{pos}}$ (unchanged) |
| Default initial width | $0.2$ | $0.75 + 0.25 \cdot 0.276 \approx 0.82$ | $`\approx 18\%`$ lower |
| Short-lived | $0.1$ | $0.75 + 0.25 \cdot 0.158 \approx 0.79$ | $`\approx 21\%`$ lower |
| Limit | $\to 0$ | $\to 0.75$ | up to $`25\%`$ lower |

**Step 3: decide.** Gaussian $i$ is cloned (if small) or split (if large) when

```math
G_i^{\text{VAD}} \ge \tau_i
```

> [!WARNING]
> **TAT only works together with VAD.** The lifespan statistics (`trbf_scale_accum`, `denom_op`) are only accumulated inside the `vad=True` branch. If you run `--tat` without `--vad`, every $\bar{\sigma}_i$ comes out as "no data", defaults to $1$, the multiplier is $1$, and **TAT silently does nothing**.

---

## 7. Training Loop: Putting It Together

For each iteration $k$:

1. Pick a random training camera; it has a timestamp $t_k$.
2. For every Gaussian compute $`\delta_i(t_k)`$ (warped if `--tow`), then $`\boldsymbol{\mu}_i(t_k)`$, $`\mathbf{q}_i(t_k)`$, $`o_i(t_k)`$.
3. Rasterise exactly like 3DGS; compute the loss $\mathcal{L}_k$ ($L_1$ + D-SSIM) against the ground-truth image and backpropagate.
4. Accumulate densification statistics (3DGS-style, or VAD/TAT-style).
5. Every 200 iterations (between `densify_from_iter` and `densify_until_iter`): clone/split, then prune Gaussians whose **peak** opacity $\mathrm{sigm}(\tilde{o}_i)$ is below `opthr`.
6. Periodically reset opacities to a small value (as in 3DGS), so useless Gaussians can be pruned.

---

## 8. Code Map

| Concept | Part | Quantity | Where in code | CLI flag |
| :--- | :--- | :--- | :--- | :--- |
| Temporal centre | Representation | $\mu_{t,i}$ | `self._trbf_center` $(N,1)$, [init](TAD-GS/scene/tad_gaussian.py#L273) | always on |
| Time offset / TOW | Representation | $\delta_i(t)$ | [`get_offset()`](TAD-GS/scene/tad_gaussian.py#L111-L125), [`offset_warping_vectorized()`](TAD-GS/scene/tad_gaussian.py#L804-L863) | `--tow` (`--temporal_span`, `--span_budget`) |
| Width scale | Representation | $\sigma_{\min}$ | [`sigma_min_for_M_frames`](TAD-GS/scene/tad_gaussian.py#L773-L784) | `--min_visible_timestamps`, `--visibility_threshold` |
| Asymmetric opacity | Representation | $o_i(t)$, $`\sigma_{i,L}, \sigma_{i,R}`$ | [`deform_gauss_op_asym()`](TAD-GS/scene/tad_gaussian.py#L188-L204), `self._trbf_scale` $(N,2)$ | default |
| Fourier motion | Representation | $\boldsymbol{\mu}_i(t)$; $\mathbf{a}, \mathbf{b}, \phi$ | [`deform_fourier_asym()`](TAD-GS/scene/tad_gaussian.py#L147-L160), `self._motion` $(N,24)$, `self._amplitude` $(N,4)$ | default |
| Rotation | Representation | $\mathbf{q}_i(t)$; $\boldsymbol{\omega}_i$ | [`get_rotation_at_t()`](TAD-GS/scene/tad_gaussian.py#L136-L139), `self._omega` $(N,4)$ | default |
| VAD | Densification | $G_i^{\text{VAD}}$ | [`d_add_densification_stats(vad=True)`](TAD-GS/scene/tad_gaussian.py#L724-L737) | `--vad` |
| TAT | Densification | $\bar{\sigma}_i$, $\tau_i$ | [`densify_gaussians()`](TAD-GS/scene/tad_gaussian.py#L698-L703), [`grad_normalizer()`](TAD-GS/scene/tad_gaussian.py#L865-L871) | `--tat` (needs `--vad`) |

**Default hyperparameters** ([`arguments/default.py`](TAD-GS/arguments/default.py#L117-L132); `coffee_martini` overrides in [`coffee_martini.json`](TAD-GS/arguments/neural_3d_video/coffee_martini.json)):

| Name | Symbol | Default | `coffee_martini` |
| :--- | :--- | :--- | :--- |
| `motion_degree` | $F$ | 4 | |
| `temporal_span` | $S$ | 10 frames | |
| `span_budget` | $B$ | 70 % | |
| `min_visible_timestamps` | $M$ | 10 | |
| `visibility_threshold` | $\tau_{\text{vis}}$ | 0.2 | |
| `beta` / `alpha` | $\beta$ / $\gamma$ | 0.75 / 0.8 | |
| `densify_grad_threshold` | $\tau_{\text{pos}}$ | 0.0002 | 0.0003 |
| `densification_interval` | | 200 | 200 |
| `densify_from_iter` → `densify_until_iter` | | 500 → 17000 | 400 → 4000 |

---

## 9. Comparison Summary

| | Static 3DGS | Deformation 4DGS (HexPlane + MLP) | Spacetime Gaussians (STG) | **TAD-GS** |
| :--- | :--- | :--- | :--- | :--- |
| **Position $\boldsymbol{\mu}_i(t)$** | constant $\boldsymbol{\mu}_{i,0}$ | $\boldsymbol{\mu}_{i,0}$ + network output | cubic polynomial in $t - \mu_{t,i}$ | Fourier series ($F=4$) in $\delta_i(t)$ |
| **Opacity $o_i(t)$** | constant | constant | symmetric bell in time | asymmetric bell ($`\sigma_{i,L} \ne \sigma_{i,R}`$), widths in units of $\sigma_{\min}$ |
| **Time input** | none | raw $t$ | plain offset $t - \mu_{t,i}$ | warped offset $\delta_i(t)$ (TOW) |
| **Gradient average** | plain mean over frustum-visible iterations | same | same (suffers ghost dilution) | opacity-weighted mean, ghosts ($o \le 0.05$) dropped (VAD) |
| **Densify threshold** | fixed $\tau_{\text{pos}}$ | fixed | fixed | per-Gaussian $\tau_i$, up to 25 % lower for short-lived Gaussians (TAT) |
| **Unseen in-between frames** (train on every $k$-th frame, test on the frames in between) | no notion of time | smooth, but blurs sudden changes | polynomials can overshoot; dynamic regions under-densified | bounded smooth trajectories and better-densified dynamic regions |
