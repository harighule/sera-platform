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

        # 3. Synaptic input (alpha-kernel weighted)
        synaptic_in = self.compute_synaptic_input()

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
