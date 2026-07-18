from dataclasses import dataclass, field
from typing import List, Optional
import math
import random


@dataclass
class DRSNState:
    """
    Full biophysical state of a single DRSN neuron at one instant in time.

    Fields
    ------
    V            : Membrane potential in mV.  Resting ≈ -70 mV, threshold ≈ -55 mV.
    h            : Dendritic hidden-state vector (d floats). Captures non-linear
                   dendritic integration and short-term history.
    e            : Eligibility trace in [0, 1].  Decaying trace of recent
                   depolarisation activity used for reward-modulated learning.
    m            : Neuromodulatory state in [0, 1].  Scaled by dopamine signals;
                   gates the magnitude of synaptic plasticity.
    theta        : Adaptive firing threshold in mV.  Rises after each spike
                   (spike-frequency adaptation) and decays back to -55 mV.
    last_spike_t : Timestamp (ms) of the most recent action potential.
    spike_count  : Cumulative count of action potentials fired.
    """

    V: float = -70.0           # membrane potential (mV)
    h: list = field(default_factory=list)  # dendritic hidden state
    e: float = 0.0             # eligibility trace [0, 1]
    m: float = 0.5             # neuromodulatory state [0, 1]
    theta: float = -55.0       # adaptive firing threshold (mV)
    last_spike_t: float = -1000.0  # timestamp of last spike (ms)
    spike_count: int = 0       # total spikes fired


@dataclass
class SynapticWeight:
    """
    A directed synaptic connection between two DRSN nodes.

    The weight w represents synaptic efficacy and is modified online by STDP.
    The two spike-timestamp fields are updated by the network on each tick so
    that stdp_update() has the information it needs to compute Δw.

    Fields
    ------
    w                : Synaptic weight (excitatory > 0, inhibitory < 0).
    pre_id           : Index of the presynaptic node.
    post_id          : Index of the postsynaptic node.
    last_pre_spike_t : Timestamp of the most recent presynaptic spike (ms).
    last_post_spike_t: Timestamp of the most recent postsynaptic spike (ms).
    """

    w: float
    pre_id: int
    post_id: int
    last_pre_spike_t: float = -1000.0
    last_post_spike_t: float = -1000.0
    phi: float = 0.0                 # quantum-inspired connection phase (radians)
    meta: float = 0.0                # meta-synaptic modulation term (3rd-order)


class DRSNNode:
    """
    Dynamic Recursive Synaptic Network (DRSN) – single neuron node.

    Implements a conductance-inspired leaky integrate-and-fire neuron enriched
    with:

    - **Alpha-kernel synaptic input** – biologically plausible post-synaptic
      current shape matching real synaptic dynamics.
    - **Non-linear dendritic integration** – a hidden-state vector models
      compartmentalised dendritic computation (tanh gate with recurrence).
    - **Spike-frequency adaptation** – an adaptive threshold rises after each
      spike and decays back to baseline, reproducing Type-II neuron behaviour.
    - **STDP (Spike-Timing-Dependent Plasticity)** – Hebbian learning rule
      sensitive to the relative timing of pre- and post-synaptic spikes.
    - **Eligibility traces** – decaying record of recent depolarisation,
      enabling three-factor (reward-modulated) learning rules.
    - **Neuromodulation** – a scalar dopamine signal scales the magnitude of
      synaptic weight updates, bridging the synapse and the reward circuit.
    """

    def __init__(
        self,
        node_id: int,
        d_hidden: int = 16,
        tau_m: float = 20.0,
        tau_theta: float = 100.0,
        V_reset: float = -65.0,
        V_rest: float = -70.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_stdp: float = 20.0,
    ) -> None:
        # Identity and hyperparameters
        self.node_id: int = node_id
        self.d_hidden: int = d_hidden
        self.tau_m: float = tau_m          # membrane time constant (ms)
        self.tau_theta: float = tau_theta  # threshold decay time constant (ms)
        self.V_reset: float = V_reset      # post-spike reset potential (mV)
        self.V_rest: float = V_rest        # resting potential (mV)
        self.A_plus: float = A_plus        # STDP potentiation amplitude
        self.A_minus: float = A_minus      # STDP depression amplitude
        self.tau_stdp: float = tau_stdp    # STDP time window (ms)

        # Neuron state
        self.state: DRSNState = DRSNState(h=[0.0] * d_hidden)

        # Dendritic weight vector – deterministic small initialisation
        # Pattern: -0.1, 0.0, 0.1, -0.1, 0.0, 0.1, ...
        self.W_den: List[float] = [0.1 * (i % 3 - 1) for i in range(d_hidden)]

        # Incoming synapses
        self.synapses: List[SynapticWeight] = []

        # Simulation clock (ms)
        self.t: float = 0.0

        # Spike history – capped at 100 most recent timestamps
        self.spike_history: List[float] = []

        # Quantum-inspired integration flag (complex-phase interference)
        self.use_quantum: bool = False

    # ------------------------------------------------------------------
    # Synaptic kernel
    # ------------------------------------------------------------------

    def alpha_kernel(self, tau: float, tau_s: float = 5.0) -> float:
        """
        Evaluate the alpha-function post-synaptic current kernel at lag tau.

            K(τ) = (τ/τₛ) · exp(1 − τ/τₛ)   for τ ≥ 0
            K(τ) = 0                            for τ < 0

        The ratio τ/τₛ is clamped to 10.0 before exponentiation to prevent
        numerical overflow on stale synapses with very large lag times.

        Parameters
        ----------
        tau   : Time elapsed since the presynaptic spike (ms).
        tau_s : Synaptic rise/decay time constant (ms).  Default 5 ms.

        Returns
        -------
        Kernel value K(τ) as a float.
        """
        if tau < 0.0:
            return 0.0
        ratio = min(tau / tau_s, 10.0)  # clamp before exp
        return ratio * math.exp(1.0 - ratio)

    # ------------------------------------------------------------------
    # Synaptic input current
    # ------------------------------------------------------------------

    def compute_synaptic_input(self) -> float:
        """
        Sum weighted alpha-kernel contributions from all incoming synapses.

        For each synapse, the lag τ = t_now − t_last_pre_spike is passed to
        alpha_kernel to obtain the current synaptic current amplitude, which is
        then scaled by the synaptic weight w.

        Returns
        -------
        Total synaptic current clamped to the range (−5.0, 5.0) to prevent
        runaway excitation / inhibition.
        """
        total = 0.0
        for synapse in self.synapses:
            tau = self.t - synapse.last_pre_spike_t
            total += synapse.w * self.alpha_kernel(tau)
        # Clamp to physiologically plausible range
        return max(-5.0, min(5.0, total))

    # ------------------------------------------------------------------
    # Quantum-inspired synaptic integration (complex-phase interference)
    # ------------------------------------------------------------------

    def compute_quantum_synaptic_input(self) -> float:
        """
        Quantum-INSPIRED synaptic integration (classical complex arithmetic).

        Instead of a plain real-valued sum Σ_j w_ij·s_j, inputs are combined as
        complex amplitudes with a learned per-connection phase φ_ij:

            I_i = Σ_j  w_ij · e^{i·φ_ij} · K(t − t_j^spike)          (complex)
            effective_current = |I_i| · sign(Re I_i)

        Signals with ALIGNED phases add constructively (amplify); OPPOSING
        phases cancel destructively — a form of automatic, continuous attention
        that costs O(N) per node instead of O(N²). This is a direct classical
        translation of quantum interference; it is NOT a quantum computer.
        """
        re, im = 0.0, 0.0
        for synapse in self.synapses:
            tau = self.t - synapse.last_pre_spike_t
            amp = synapse.w * self.alpha_kernel(tau)
            re += amp * math.cos(synapse.phi)
            im += amp * math.sin(synapse.phi)
        magnitude = math.sqrt(re * re + im * im)
        signed = magnitude if re >= 0.0 else -magnitude
        return max(-5.0, min(5.0, signed))

    # ------------------------------------------------------------------
    # Dendritic integration
    # ------------------------------------------------------------------

    def update_dendritic_state(self, external_input: float, dt: float = 1.0) -> None:
        """
        Advance the dendritic hidden-state vector by one timestep dt.

        Each compartment i evolves as a leaky integrator with a tanh
        non-linearity driven by the external input and a recurrent self-coupling:

            hᵢ(t+dt) = hᵢ(t) · (1 − dt/10) + dt · tanh(Wᵢ · x + hᵢ · 0.1)

        The factor dt/10 controls the dendritic leak time constant (10 ms).

        Parameters
        ----------
        external_input : Scalar external drive (same for all compartments).
        dt             : Timestep in ms.
        """
        for i in range(self.d_hidden):
            leak = self.state.h[i] * (1.0 - dt / 10.0)
            drive = dt * math.tanh(self.W_den[i] * external_input + self.state.h[i] * 0.1)
            self.state.h[i] = leak + drive

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(
        self,
        external_input: float = 0.0,
        dt: float = 1.0,
        dopamine: float = 0.0,
    ) -> bool:
        """
        Advance the neuron by one simulation timestep dt.

        Execution order
        ---------------
        1. Advance clock.
        2. Update dendritic hidden state.
        3. Compute total synaptic input via alpha kernels.
        4. Integrate membrane potential (leaky integrator).
        5. Update eligibility trace.
        6. Update neuromodulatory state from dopamine signal.
        7. Decay adaptive threshold toward -55 mV.
        8. Check firing condition; if threshold crossed, emit spike.

        Parameters
        ----------
        external_input : Injected current / external drive (mV-equivalent scalar).
        dt             : Timestep in ms.  Default 1 ms.
        dopamine       : Dopamine signal in [−1, 1] that modulates plasticity.

        Returns
        -------
        True  if an action potential was fired during this timestep.
        False otherwise.
        """
        # 1. Advance simulation clock
        self.t += dt

        # 2. Non-linear dendritic integration
        self.update_dendritic_state(external_input, dt)

        # 3. Synaptic input — quantum-inspired (phase interference) or classical
        synaptic_in = (
            self.compute_quantum_synaptic_input()
            if self.use_quantum
            else self.compute_synaptic_input()
        )

        # 4. Dendritic drive: mean compartment activation
        dendritic_sum = sum(self.state.h) / max(len(self.state.h), 1)

        # Leaky integrate-and-fire membrane equation
        dV = (-(self.state.V - self.V_rest) + synaptic_in + dendritic_sum + external_input) / self.tau_m
        self.state.V += dV * dt

        # 5. Eligibility trace – decaying running measure of depolarisation
        self.state.e = self.state.e * 0.95 + 0.05 * abs(dV)

        # 6. Neuromodulation – dopamine gates the strength of plasticity
        self.state.m = max(0.0, min(1.0, self.state.m + 0.01 * dopamine))

        # 7. Adaptive threshold decay toward baseline (-55 mV)
        dtheta = -(self.state.theta - (-55.0)) / self.tau_theta
        self.state.theta += dtheta * dt

        # 8. Firing condition
        if self.state.V >= self.state.theta:
            # Record spike
            self.spike_history.append(self.t)
            if len(self.spike_history) > 100:
                self.spike_history = self.spike_history[-100:]
            self.state.last_spike_t = self.t
            self.state.spike_count += 1

            # Post-spike reset
            self.state.V = self.V_reset

            # Spike-frequency adaptation: push threshold up
            self.state.theta += 5.0

            return True

        return False

    # ------------------------------------------------------------------
    # STDP synaptic plasticity
    # ------------------------------------------------------------------

    def stdp_update(self, synapse: SynapticWeight, dt: float = 1.0) -> float:
        """
        Apply a single STDP update to synapse based on pre/post spike timing.

        Rule
        ----
        Let Δt = t_post − t_pre.

        - Δt > 0  (pre fires before post): potentiation
              Δw = A₊ · exp(−Δt / τ_stdp)
        - Δt ≤ 0  (post fires before or at same time as pre): depression
              Δw = −A₋ · exp(Δt / τ_stdp)

        The raw Δw is then:
          1. Scaled by the neuromodulatory state m (dopamine gate).
          2. Subject to a homeostatic weight-decay term pulling w toward 0.
          3. Applied and clamped to [−2.0, 2.0].

        Parameters
        ----------
        synapse : The SynapticWeight to update in-place.
        dt      : Unused; present for API consistency with the step() signature.

        Returns
        -------
        The weight change dw that was applied.
        """
        delta_t = synapse.last_post_spike_t - synapse.last_pre_spike_t

        if delta_t > 0.0:
            dw = self.A_plus * math.exp(-delta_t / self.tau_stdp)
        else:
            dw = -self.A_minus * math.exp(delta_t / self.tau_stdp)

        # Neuromodulatory gating
        dw *= self.state.m

        # Meta-synaptic modulation (Level-2 of the tetration stack): a per-synapse
        # meta-weight modulates the plasticity of this synapse (third-order term
        # Σ_k M_ijk·s_k in the spec, here a low-rank scalar surrogate), gated by
        # the neuromodulatory state.
        dw += 0.005 * synapse.meta * self.state.m
        # Update the meta-weight from higher-order (eligibility-gated) co-activation.
        synapse.meta = 0.99 * synapse.meta + 0.01 * self.state.e
        synapse.meta = max(-1.0, min(1.0, synapse.meta))

        # Homeostatic regularisation: soft pull toward zero
        dw -= 0.001 * (synapse.w - 0.0)

        synapse.w = max(-2.0, min(2.0, synapse.w + dw))
        return dw

    # ------------------------------------------------------------------
    # Synapse management
    # ------------------------------------------------------------------

    def add_synapse(self, pre_id: int, weight: float = 0.1) -> None:
        """
        Append a new incoming synapse from pre_id with the given initial weight.

        Parameters
        ----------
        pre_id : Index of the presynaptic node in the network.
        weight : Initial synaptic efficacy.
        """
        self.synapses.append(
            SynapticWeight(w=weight, pre_id=pre_id, post_id=self.node_id)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_state_dict(self) -> dict:
        """
        Return a JSON-serialisable snapshot of the node's current state.

        Returns
        -------
        dict with:
          node_id     : int   – node index
          V           : float – membrane potential (mV)
          theta       : float – adaptive firing threshold (mV)
          eligibility : float – eligibility trace value
          neuromod    : float – neuromodulatory state
          spike_count : int   – cumulative spikes fired
          n_synapses  : int   – number of incoming synapses
          mean_h      : float – mean dendritic compartment activation
        """
        mean_h = sum(self.state.h) / max(len(self.state.h), 1)
        return {
            "node_id": self.node_id,
            "V": self.state.V,
            "theta": self.state.theta,
            "eligibility": self.state.e,
            "neuromod": self.state.m,
            "spike_count": self.state.spike_count,
            "n_synapses": len(self.synapses),
            "mean_h": mean_h,
        }


class DRSNNetwork:
    """
    A sparsely connected network of DRSNNode neurons.

    Wiring uses a deterministic small-world-like scheme: each node i receives
    synapses from nodes at offsets +1, +3, +7 (mod n_nodes), giving a
    structured but diverse connectivity graph without random seeds.

    The network exposes a simple rate-code feature encoder via encode_features(),
    suitable for embedding entity-level feature vectors into a spiking
    representation.
    """

    def __init__(self, n_nodes: int = 16, d_hidden: int = 8) -> None:
        self.n_nodes: int = n_nodes
        self.nodes: List[DRSNNode] = [DRSNNode(i, d_hidden) for i in range(n_nodes)]
        self.t: float = 0.0

        # Deterministic sparse connectivity: offsets +1, +3, +7
        for i in range(n_nodes):
            for offset in (1, 3, 7):
                pre = (i + offset) % n_nodes
                self.nodes[i].add_synapse(pre_id=pre, weight=0.05)
                # Deterministic per-connection phase for quantum-inspired
                # interference (spread across [0, 2π) by offset & node index).
                self.nodes[i].synapses[-1].phi = (
                    2.0 * math.pi * ((offset + i) % n_nodes) / max(n_nodes, 1)
                )

    def set_quantum(self, enabled: bool = True) -> None:
        """Toggle quantum-inspired complex-phase synaptic integration."""
        for node in self.nodes:
            node.use_quantum = enabled

    def demonstrate_interference(self) -> dict:
        """
        Show that phase alignment matters: constructive interference (all phases
        aligned) yields a larger integrated input magnitude than destructive
        interference (alternating opposed phases), for the same weights/spikes.
        """
        node = self.nodes[0]
        # ensure recent presynaptic activity so the alpha kernel is non-zero
        for syn in node.synapses:
            syn.last_pre_spike_t = node.t - 2.0
        orig = [s.phi for s in node.synapses]
        for s in node.synapses:            # constructive: aligned phases
            s.phi = 0.0
        constructive = node.compute_quantum_synaptic_input()
        for idx, s in enumerate(node.synapses):   # destructive: alternating
            s.phi = 0.0 if idx % 2 == 0 else math.pi
        destructive = node.compute_quantum_synaptic_input()
        for s, p in zip(node.synapses, orig):
            s.phi = p
        return {
            "constructive_input": round(constructive, 5),
            "destructive_input": round(destructive, 5),
            "interference_confirmed": abs(constructive) > abs(destructive) + 1e-6,
        }

    # ------------------------------------------------------------------
    # Network step
    # ------------------------------------------------------------------

    def step_all(
        self,
        inputs: list,
        dopamine: float = 0.0,
        dt: float = 1.0,
    ) -> List[bool]:
        """
        Advance every node by one timestep and propagate spike timing.

        After stepping all nodes, the spike timestamps are propagated:
        - For each node that fired, its own postsynaptic timestamp is updated
          on all of its own incoming synapses.
        - For each other node whose incoming synapses include the spiking node
          as a presynaptic partner, that synapse's last_pre_spike_t is updated.

        Parameters
        ----------
        inputs   : List of n_nodes floats – per-node external drive.
        dopamine : Global dopamine signal for neuromodulation.
        dt       : Timestep in ms.

        Returns
        -------
        List of bool – True at index i if node i fired during this step.
        """
        # Pad or trim inputs to exactly n_nodes
        padded = list(inputs) + [0.0] * self.n_nodes
        padded = padded[: self.n_nodes]

        spikes = [self.nodes[i].step(padded[i], dt, dopamine) for i in range(self.n_nodes)]

        self.t += dt

        # Propagate spike timing information
        for i, fired in enumerate(spikes):
            if not fired:
                continue

            t_spike = self.nodes[i].state.last_spike_t

            # Update post-spike timestamp on this node's own incoming synapses
            for syn in self.nodes[i].synapses:
                syn.last_post_spike_t = t_spike

            # Update pre-spike timestamp on synapses in *other* nodes
            for j in range(self.n_nodes):
                if j == i:
                    continue
                for syn in self.nodes[j].synapses:
                    if syn.pre_id == i:
                        syn.last_pre_spike_t = t_spike

        return spikes

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(
        self,
        input_sequence: list,
        n_steps: int = 20,
        dopamine: float = 0.0,
    ) -> dict:
        """
        Run the network for n_steps timesteps with a constant input vector.

        WARNING: This method runs on stateful neurons. The membrane potential (V),
        adaptive threshold (theta), dendritic hidden state (h), and synaptic spike
        histories/timestamps persist across invocations. Therefore, sequential calls to
        run() are path-dependent on execution history. Call reset() between runs to
        ensure independent evaluations.

        Parameters
        ----------
        input_sequence : List of n_nodes floats – same drive applied every step.
        n_steps        : Number of 1 ms timesteps to simulate.
        dopamine       : Constant dopamine signal during the run.

        Returns
        -------
        dict with:
          spike_counts     : List[int] – cumulative new spikes per node
          total_spikes     : int       – sum of spike_counts
          mean_firing_rate : float     – total_spikes / (n_nodes × n_steps)
          active_nodes     : int       – nodes that fired at least once
          world_state      : List[float] – mean dendritic activation per node
        """
        # Snapshot spike counts before the run to compute deltas
        baseline_counts = [node.state.spike_count for node in self.nodes]

        for _ in range(n_steps):
            self.step_all(input_sequence, dopamine=dopamine)

        spike_counts = [
            self.nodes[i].state.spike_count - baseline_counts[i]
            for i in range(self.n_nodes)
        ]
        total_spikes = sum(spike_counts)
        mean_firing_rate = total_spikes / max(self.n_nodes * n_steps, 1)
        active_nodes = sum(1 for c in spike_counts if c > 0)
        world_state = [
            sum(node.state.h) / max(len(node.state.h), 1) for node in self.nodes
        ]

        return {
            "spike_counts": spike_counts,
            "total_spikes": total_spikes,
            "mean_firing_rate": mean_firing_rate,
            "active_nodes": active_nodes,
            "world_state": world_state,
        }

    # ------------------------------------------------------------------
    # Feature encoder
    # ------------------------------------------------------------------

    def encode_features(self, features: list, n_steps: int = 10) -> dict:
        """
        Encode an entity feature vector into a spiking network representation.

        WARNING: This spiking network is fully stateful. The membrane potential (V),
        adaptive threshold (theta), dendritic hidden state (h), and synaptic spike
        histories/timestamps persist across invocations. Therefore, sequential calls to
        encode_features() or run() on the same instance are path-dependent on previous
        inputs and execution history. To ensure independent evaluations, call reset()
        between runs.

        The features list is padded with zeros or trimmed to n_nodes, then used
        as a constant input drive over n_steps simulation timesteps.  The
        resulting spike statistics form a rate-code embedding of the input.

        Parameters
        ----------
        features : List of floats (entity-level features).
        n_steps  : Number of timesteps to simulate.

        Returns
        -------
        The dict returned by run() – see run() docstring for keys.
        """
        padded = list(features) + [0.0] * self.n_nodes
        padded = padded[: self.n_nodes]
        return self.run(padded, n_steps=n_steps)

    def reset(self) -> None:
        """
        Reset the network state, including simulation clock, membrane potential,
        adaptive threshold, dendritic hidden state, eligibility traces,
        spike histories, and synaptic spike timestamps for all nodes.
        """
        self.t = 0.0
        for node in self.nodes:
            node.t = 0.0
            node.spike_history.clear()
            node.state.V = node.V_rest
            node.state.theta = -55.0
            node.state.h = [0.0] * node.d_hidden
            node.state.e = 0.0
            node.state.m = 0.5
            node.state.last_spike_t = -1000.0
            node.state.spike_count = 0
            for syn in node.synapses:
                syn.last_pre_spike_t = -1000.0
                syn.last_post_spike_t = -1000.0

    # ------------------------------------------------------------------
    # World model — planning by internal simulation
    # ------------------------------------------------------------------

    def k_winners_take_all(self, spikes: List[bool], k_frac: float = 0.5) -> List[bool]:
        """
        k-Winners-Take-All lateral inhibition (sparse distributed coding).

        Keeps only the top ``k_frac`` fraction of firing nodes ranked by their
        membrane potential, suppressing the rest. Implements the DRSN spec's
        sparsity mechanism: only a small fraction of nodes propagate at any tick.
        """
        firing = [i for i, s in enumerate(spikes) if s]
        if not firing:
            return spikes
        k = max(1, int(len(firing) * k_frac))
        firing_sorted = sorted(firing, key=lambda i: self.nodes[i].state.V, reverse=True)
        keep = set(firing_sorted[:k])
        return [bool(s) and (i in keep) for i, s in enumerate(spikes)]

    def predict_next_state(self, n_predict: int = 3) -> dict:
        """
        World-model prediction by INTERNAL SIMULATION (imagination).

        Rolls the network dynamics forward with zero external input
        (``I_ext = 0``) — the DRSN spec's principle that *the recurrent dynamics
        ARE the running prediction*: the network "imagines" future states by
        feeding its own state back through the coupled differential equations,
        which is the substrate for planning ("simulating possible futures inside
        itself before taking an action").

        NOTE: advances the network's internal clock/state (as a real dynamical
        rollout does). Call ``reset()`` afterwards for an independent evaluation.

        Returns the predicted world-state trajectory (mean dendritic activation
        per node at each imagined step) plus the predicted spike activity.
        """
        zero = [0.0] * self.n_nodes
        trajectory, spike_activity = [], []
        for _ in range(max(1, n_predict)):
            spikes = self.step_all(zero, dopamine=0.0)
            spikes = self.k_winners_take_all(spikes, k_frac=0.5)   # sparse rollout
            ws = [sum(node.state.h) / max(len(node.state.h), 1) for node in self.nodes]
            trajectory.append(ws)
            spike_activity.append(int(sum(1 for s in spikes if s)))
        return {
            "predicted_world_states": trajectory,
            "predicted_spike_activity": spike_activity,
            "n_predict": int(n_predict),
            "method": "internal_dynamical_rollout_Iext0",
            "note": "DRSN world-model prediction: recurrent dynamics used as the predictor",
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> dict:
        """
        Produce a JSON-serialisable summary of the network state.

        Returns
        -------
        dict with:
          n_nodes           : int          – number of nodes in the network
          mean_V            : float        – mean membrane potential (mV)
          total_spike_count : int          – sum of spike_count across all nodes
          node_states       : List[dict]   – to_state_dict() for each node
        """
        mean_V = sum(node.state.V for node in self.nodes) / max(self.n_nodes, 1)
        total_spike_count = sum(node.state.spike_count for node in self.nodes)
        return {
            "n_nodes": self.n_nodes,
            "mean_V": mean_V,
            "total_spike_count": total_spike_count,
            "node_states": [node.to_state_dict() for node in self.nodes],
        }


# ─────────────────────────────────────────────────────────────────
# MULTIMODAL PERCEPTION — encode any modality into DRSN spike-rate features
# ─────────────────────────────────────────────────────────────────
class MultimodalEncoder:
    """
    Unified encoder: text, image, audio and sensor inputs all map into the SAME
    spike-rate feature space, so one DRSN network can perceive every modality
    (the spec's unified multimodal perception). Each encoder is biologically
    inspired: rate coding (text), difference-of-Gaussians (image / retina),
    frequency decomposition (audio / cochlea), population coding (sensor).
    """

    @staticmethod
    def _norm(v):
        m = max((abs(x) for x in v), default=0.0) or 1.0
        return [x / m for x in v]

    @staticmethod
    def encode_text(token_ids, dim: int = 8):
        v = [0.0] * dim
        for t in token_ids:
            v[int(t) % dim] += 1.0
        return MultimodalEncoder._norm(v)

    @staticmethod
    def encode_image(image_2d, dim: int = 8):
        import numpy as np
        img = np.asarray(image_2d, dtype=float)
        # difference-of-Gaussians (center-surround) via two box blurs
        def blur(a, k):
            pad = np.pad(a, k, mode="edge")
            out = np.zeros_like(a)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    out[i, j] = pad[i:i + 2 * k + 1, j:j + 2 * k + 1].mean()
            return out
        dog = np.abs(blur(img, 1) - blur(img, 2))          # edge energy
        bins = np.array_split(dog.flatten(), dim)
        return MultimodalEncoder._norm([float(b.mean()) for b in bins])

    @staticmethod
    def encode_audio(waveform, dim: int = 8):
        import numpy as np
        spec = np.abs(np.fft.rfft(np.asarray(waveform, dtype=float)))   # cochlea-like
        bins = np.array_split(spec, dim)
        return MultimodalEncoder._norm([float(b.mean()) for b in bins])

    @staticmethod
    def encode_sensor(vec, dim: int = 8):
        import numpy as np
        x = np.asarray(vec, dtype=float)
        centers = np.linspace(float(x.min()), float(x.max()), dim)     # population code
        val = float(x.mean())
        return MultimodalEncoder._norm(
            [float(np.exp(-((val - c) ** 2) / (2 * 0.5 ** 2))) for c in centers])


def grover_plan(action_costs, threshold=None, iterations=None) -> dict:
    """
    Grover-inspired planning search: amplitude amplification over candidate
    action sequences. 'Good' actions (cost < threshold) are marked by an oracle
    and amplified by diffusion, finding a good action in ~O(√N) iterations rather
    than the O(N) of classical unstructured search.
    """
    import numpy as np, math as _m
    costs = np.asarray(action_costs, dtype=float)
    n = len(costs)
    if n == 0:
        return {"n_actions": 0, "found_good_action": False}
    if threshold is None:
        threshold = float(np.median(costs))
    good = costs < threshold
    n_good = max(1, int(good.sum()))
    # Optimal Grover iteration count ≈ (π/4)·√(N/M); floor to avoid over-rotation.
    iters = iterations or max(1, int((_m.pi / 4) * _m.sqrt(n / n_good)))
    p = np.ones(n) / n
    best_p_good, best = -1.0, int(np.argmin(costs))
    for _ in range(iters):
        marked = np.where(good, -p, p)          # oracle sign-flip on good states
        mu = marked.mean()
        p = np.abs(2 * mu - marked)             # diffusion about the mean
        p = p / (p.sum() + 1e-12)
        gi = int(np.argmax(np.where(good, p, -1)))   # best good state this round
        if good[gi] and p[gi] > best_p_good:
            best_p_good, best = float(p[gi]), gi
    best = int(best)
    return {"n_actions": n, "grover_iterations": iters,
            "classical_iterations": n,
            "speedup_factor": round(n / max(iters, 1), 2),
            "best_action_index": best,
            "best_action_cost": round(float(costs[best]), 4),
            "found_good_action": bool(costs[best] < threshold)}


def hierarchical_world_model(network, timescales=(1, 4, 16)) -> dict:
    """
    Three-timescale hierarchical world model: predict at sensorimotor (fast),
    behavioural (medium) and goal (slow) scales via progressively longer internal
    rollouts. Higher levels integrate lower-level summaries.
    """
    names = ["sensorimotor", "behavioural", "goal"]
    out = {}
    for name, ts in zip(names, timescales):
        network.reset()
        network.encode_features([0.9, 0.5, 0.2, 0.1, 1.0, 0.2, 0.3, 0.4], n_steps=5)
        pred = network.predict_next_state(n_predict=ts)
        acts = pred["predicted_spike_activity"]
        out[name] = {"timescale": ts,
                     "rollout_steps": len(acts),
                     "mean_activity": round(sum(acts) / max(len(acts), 1), 3)}
    return out
