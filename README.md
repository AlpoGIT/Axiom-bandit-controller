(This baby "research project" was framed and solved with Microsoft Copilot... and some feedback from an actual human). I'm not an expert in AXIOM algorithm, but the code seems to work.
Here are its result (described by itself). I find it interesting that it help framing a specific problem, and help with the coding part.


# axiom-bandit-controller

An adaptive agent based on Active Inference principles, capable of aligning long-term behavior with internal goals—even when no single action satisfies them. Inspired by the AXIOM framework, this implementation introduces a minimal trajectory-aware control loop to regulate cumulative reward toward a target value through dynamic policy mixing.

## 🌐 Motivation

Standard AXIOM agents select actions by minimizing expected free energy (EFE), seeking arms that match their internal goal (e.g. reward ≈ 0.5). However, when no arm delivers the target directly, traditional inference collapses onto the "least-bad" option, leaving the agent unable to regulate its long-term outcome.

This project solves that limitation.

## 🧠 Key Insight

Modify the agent’s behavior with a single control term:
```python
pragmatic = (self.internal_goal - np.mean(self.all_rewards)) * self.means
