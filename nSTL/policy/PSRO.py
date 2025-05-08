import torch
import numpy as np
import cvxpy as cp
from config.team_config import ConfigTeam, generate_start_end_positions
from env.dynamic_env import MultiAgentEnv, batched_rollout
from robust.team_stl_helper import STLFormulaReachAvoidTeam
from policy.team_modules_helper import make_joint_policy, JointPolicy
from policy.Oracle import JointSTLOracle
from env.dynamic_env import *
import wandb
from tqdm import trange

class ZeroSumMetaSolver:
    """
    Meta-solver for two-team zero-sum restricted games.
    Supports two methods: 'LP' for CVXPY-based exact Nash, 'FP' for fictitious play.
    """
    def __init__(self, method: str = 'LP', fp_iters: int = 50):
        if method == 'LP' and cp is None:
            raise ImportError('cvxpy is required for LP solver but not installed')
        self.method = method
        self.fp_iters = fp_iters

    def solve(self, payoff):
        """
        Solve for meta-strategies mu_A, mu_B and exploitability.
        Arguments:
          payoff: array-like of shape (nA, nB).
        Returns:
          mu_A: numpy array, shape (nA,)
          mu_B: numpy array, shape (nB,)
          exploit: float exploitability
        """
        # ensure numpy matrix
        M = payoff.detach().cpu().numpy() if hasattr(payoff, 'detach') else np.array(payoff)
        nA, nB = M.shape

        if self.method == 'LP':
            # row-player LP: maximize v s.t. M.T @ muA >= v, sum(muA)=1, muA>=0
            muA = cp.Variable(nA)
            v = cp.Variable()
            constraints = [muA >= 0, cp.sum(muA) == 1, M.T @ muA >= v]
            prob = cp.Problem(cp.Maximize(v), constraints)
            prob.solve()
            muA_val = muA.value
            v_val = v.value
            # col-player LP: minimize w s.t. M @ muB <= w, sum(muB)=1, muB>=0
            muB = cp.Variable(nB)
            w = cp.Variable()
            constraints2 = [muB >= 0, cp.sum(muB) == 1, M @ muB <= w]
            prob2 = cp.Problem(cp.Minimize(w), constraints2)
            prob2.solve()
            muB_val = muB.value
            w_val = w.value
            # At equilibrium v <= w; exploitability is the gap w - v ≥ 0
            exploit = float(w_val - v_val)
            if exploit < -1e-9:
                raise ValueError(
                    f"Exploitability became negative ({exploit:.4e}) in LP solver; "
                    "check payoff sign conventions."
                )
            exploit = max(exploit, 0.0)
            return muA_val, muB_val, exploit

        elif self.method == 'FP':
            # simple two-player fictitious play
            muA = np.ones(nA) / nA
            muB = np.ones(nB) / nB
            brA = 0
            brB = 0
            for t in range(self.fp_iters):
                # best response to opponent mix
                valsA = M.dot(muB)
                brA = int(np.argmax(valsA))
                valsB = M.T.dot(muA)
                brB = int(np.argmin(valsB))
                # update empirical distributions
                eA = np.zeros(nA); eA[brA] = 1
                eB = np.zeros(nB); eB[brB] = 1
                muA = (muA * t + eA) / (t + 1)
                muB = (muB * t + eB) / (t + 1)
            # compute exploitability
            val = muA.dot(M).dot(muB)
            brA_val = M[brA, :].dot(muB)
            brB_val = muA.dot(M[:, brB])
            exploit = float((brA_val - val) + (val - brB_val))
            return muA, muB, exploit

        else:
            raise ValueError(f"Unknown meta-solver method: {self.method}")

class PSRODriver:
    """
    Policy Space Response Oracles Driver for N player Team STL Games.
    """
    def __init__(self, cfg: ConfigTeam, dynamics_fn, state_dim, ctrl_dim, team_size=2):
        self.cfg = cfg
        wandb.init(project="PSRO_Team_STL", config={"fsp_iteration": cfg.fsp_iteration})
        # Build environment
        self.env = MultiAgentEnv(
            state_dim=state_dim,
            control_dim=ctrl_dim,
            n_agents=team_size * 2,
            dynamic_func=dynamics_fn
        ) 
        # Build STL formula
        obs = cfg.get_obstacles()
        self.stl = STLFormulaReachAvoidTeam(
            obs_boxes=[obs[0], obs[1]],
            circle_obs=obs[2],
            goal_box=obs[4],
            T=cfg.T,
            safe_d=cfg.u_max
        )
        # Initialize populations with one random joint policy each
        self.pop_A = [make_joint_policy(team_size, state_dim, ctrl_dim, cfg.total_time_step, cfg.device, "A")]
        self.pop_B = [make_joint_policy(team_size, state_dim, ctrl_dim, cfg.total_time_step, cfg.device, "B")]
        # Ensure policy modules are on the correct device
        for policy in self.pop_A:
            policy.to(self.cfg.device)
        for policy in self.pop_B:
            policy.to(self.cfg.device)
        # Oracle and meta-solver
        self.oracle = JointSTLOracle(cfg=cfg, env=self.env, team_size=team_size, stl_formula=self.stl, total_t=cfg.T)
        self.solver = ZeroSumMetaSolver()
        self.payoff_cache = {}
        self.mu_A = np.array([1.0])
        self.mu_B = np.array([1.0])

    def _build_payoff_matrix(self):
        nA, nB = len(self.pop_A), len(self.pop_B)
        payoff = torch.zeros(nA, nB, device=self.cfg.device)
        for i, pi in enumerate(self.pop_A):
            for j, pj in enumerate(self.pop_B):
                if (i, j) not in self.payoff_cache:
                    r = self._rollout_robustness(pi, pj)  # one rollout only
                    self.payoff_cache[(i, j)] = r
                payoff[i, j] = self.payoff_cache[(i, j)]
        max_payoff = torch.max(payoff).item()
        wandb.log({"Max Robustness": max_payoff})
        return payoff

    # ------------------------------------------------------------------ #
    # helper: one rollout → robustness                                   #
    # ------------------------------------------------------------------ #
    def _rollout_robustness(self, pol_A: JointPolicy, pol_B: JointPolicy):
        """
        Roll out a single episode between joint policies pol_A (row player)
        and pol_B (col player) and return the mean STL robustness of team A.

        The rollout is batched (batch size = cfg.batch_size) to reduce variance.
        No gradients are recorded.
        """
        env = self.env
        B = self.cfg.batch_size
        device = self.cfg.device
        total_sd = env.n_agents * env.state_dim

        with torch.no_grad():
            # random initial states in [-1, 1]
            init = torch.rand(B, total_sd, device=device) * 2.0 - 1.0
            env.reset(init)

            traj = batched_rollout(env, pol_A, pol_B, T=self.cfg.total_time_step)

            sd = env.state_dim
            pos_dim = 2  # first two dims are (x, y) positions
            # slice trajectories into per‑agent (x,y) traces
            ego_trajs = [traj[:, :, k * sd : k * sd + pos_dim]
                         for k in range(len(pol_A.policies))]
            opp_offset = len(pol_A.policies)
            opp_trajs = [traj[:, :, (opp_offset + k) * sd : (opp_offset + k) * sd + pos_dim]
                         for k in range(len(pol_B.policies))]

            rob_ego = self.stl.compute_robustness_ego(ego_trajs, opp_trajs)
            rob_opp = -rob_ego
            wandb.log({"Rob_ego": rob_ego.mean().item(),
                       "Rob_opp": rob_opp.mean().item()},
                       commit=False)
            return rob_ego.mean().item()
    
    def iterate(self):
        # 1) Build payoff
        payoff = self._build_payoff_matrix()
        self.mu_A, self.mu_B, exploit = self.solver.solve(payoff)
        # pass current μ to the oracle
        brA = self.oracle.train("A", self.pop_B, self.mu_B)
        brB = self.oracle.train("B", self.pop_A, self.mu_A)
        # 4) Add new policies
        self.pop_A.append(brA)
        self.pop_B.append(brB)
        return exploit

    def run(self, iterations=None):
        iters = iterations or self.cfg.fsp_iteration
        wandb.config.update({"fsp_iteration": iters})
        for k in trange(iters, desc="PSRO loop"):
            exploit = self.iterate()
            print(f"[PSRO] Iter {k+1}/{iters} exploitability = {exploit:.4f}")
            wandb.log({'exploitability': exploit, 'iteration': k+1})


if __name__ == "__main__":
    # 2-vs-2 random start/goal pairs
    ego_choices = [generate_start_end_positions() for _ in range(2)]
    opp_choices = [generate_start_end_positions() for _ in range(2)]
    print(f"=======Start Positions : EGO : {ego_choices}========")
    print(f"=======Start Positions : OPP : {opp_choices}========")

    # Configuration
    cfg = ConfigTeam(ego_choices, opp_choices)

    # Initialize PSRO driver for single-integrator dynamics
    driver = PSRODriver(
        cfg=cfg,
        dynamics_fn=single_integrator,
        state_dim=2,
        ctrl_dim=2,
        team_size=2
    )

    # Run PSRO loop
    driver.run()