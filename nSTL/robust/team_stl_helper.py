# =======================================================
#   Generic STL primitives for m v n team games
#   (drop this into a new utils file, e.g. stl_generic.py)
# =======================================================

from robust.stlcg import stlcg
import torch
from typing import Tuple, List
from nSTL.robust.stl_helper import *

# -------------------------------------------------------
# Helper: create an stlcg.Expression for a constant tensor
# -------------------------------------------------------
def _const(name: str, tensor: torch.Tensor) -> stlcg.Expression:
    """Wrap a constant tensor into an STLCG Expression."""
    return stlcg.Expression(name, tensor.unsqueeze(-1).unsqueeze(-1))


# -------------------------------------------------------
# 1) Inside-box predicate for arbitrary k-D state
# -------------------------------------------------------
def inside_box_kd(x: torch.Tensor,
                  bounds: torch.Tensor,
                  name_prefix: str = "x") -> Tuple[stlcg.STL_Formula, Tuple]:
    """
    x       : (T, k) trajectory  (T = time length)
    bounds  : (2*k,) tensor  [low₁, high₁, …, low_k, high_k]
    Returns (predicate, inputs) suitable for .robustness().
    """
    k = x.shape[1]
    assert bounds.numel() == 2 * k, "bounds must have 2*k elements"

    # Build conjunction over all dimensions
    preds = []
    inputs = []
    for d in range(k):
        low  = _const(f"{name_prefix}{d}_low",  bounds[2*d:2*d+1])
        high = _const(f"{name_prefix}{d}_high", bounds[2*d+1:2*d+2])
        xd   = stlcg.Expression(f"{name_prefix}{d}", x[:, d:d+1].unsqueeze(0))
        preds.append((xd > low) & (xd < high))
        inputs.append((xd, xd))

    inside = preds[0]
    for p in preds[1:]:
        inside = inside & p
    return inside, tuple(inputs)


# -------------------------------------------------------
# 2) Distance-greater / -less predicates (collision & safety)
# -------------------------------------------------------
def distance_greater(x: torch.Tensor,
                     y: torch.Tensor,
                     safe_d: torch.Tensor,
                     name: str = "dist") -> Tuple[stlcg.STL_Formula, stlcg.Expression]:
    """
    x, y    : (T, k) trajectories of two agents
    safe_d  : scalar tensor  minimal safe distance
    Returns (predicate ‘‖x−y‖₂ > safe_d’, distance Expression)
    """
    d_expr = stlcg.Expression(name, torch.norm(x - y.unsqueeze(0), dim=-1,
                                               keepdim=True).unsqueeze(0))
    r_expr = _const("d_min", safe_d)
    return (d_expr > r_expr), d_expr


def distance_less_equal(x: torch.Tensor,
                        y: torch.Tensor,
                        thr_d: torch.Tensor,
                        name: str = "dist") -> Tuple[stlcg.STL_Formula, stlcg.Expression]:
    """
    Convenience predicate for opponents that *want* collision.
    """
    d_expr = stlcg.Expression(name, torch.norm(x - y.unsqueeze(0), dim=-1,
                                               keepdim=True).unsqueeze(0))
    r_expr = _const("d_thr", thr_d)
    return (d_expr <= r_expr), d_expr


# -------------------------------------------------------
# 3) Always / Eventually wrappers for safe distance
# -------------------------------------------------------
def always_safe_distance(x: torch.Tensor,
                         y: torch.Tensor,
                         safe_d: torch.Tensor,
                         horizon: List[int]) -> Tuple[stlcg.STL_Formula, stlcg.Expression]:
    """
    horizon : [t_start, t_end] interval for the □ operator
    Example:   always_safe_distance(qE, qOj, d_min, [0, T])
    """
    pred, d_expr = distance_greater(x, y, safe_d)
    return stlcg.Always(subformula=pred, interval=horizon), d_expr


def eventually_collision(x: torch.Tensor,
                         y: torch.Tensor,
                         thr_d: torch.Tensor,
                         horizon: List[int]) -> Tuple[stlcg.STL_Formula, stlcg.Expression]:
    """
    Opponent objective: ‘eventually within thr_d’.
    """
    pred, d_expr = distance_less_equal(x, y, thr_d)
    return stlcg.Eventually(subformula=pred, interval=horizon), d_expr


# ==============================================================
#  General-purpose reach-avoid STL for  m  ego agents  vs  n  opp
#  Works in 2-D; extend predicates to 3-D analogously.
#  Drop this into e.g.  utils/stl_reach_avoid_team.py
# ==============================================================

class STLFormulaReachAvoidTeam:
    """
    * Ego team (size = m) must:
        – eventually stay inside BOTH obs_1 & obs_2 for Δ steps,
        – always stay outside circular obstacle obs_3,
        – always avoid collisions with ANY opponent.
    * Opponent team (size = n) payoff is the logical negation.

    Parameters
    ----------
    obs_boxes : list[Tensor]        – [obs_1, obs_2] each (4,) (x1,x2,y1,y2)
    circle_obs: Tensor             – (3,)  (xc, yc, radius)
    goal_box  : Tensor             – (4,)  (x1,x2,y1,y2)  (if needed for extra tasks)
    T         : int                – horizon for □/◇
    safe_d    : float or Tensor(1) – minimal separation

    Note
    ----
    * The class is **modular**: just pass lists of trajectories
      whose lengths define m and n at run-time.
    """

    def __init__(self,
                 obs_boxes,
                 circle_obs,
                 goal_box,
                 T: int,
                 safe_d: torch.Tensor):

        super().__init__()
        self.obs_1, self.obs_2 = obs_boxes
        self.circle = circle_obs
        self.goal   = goal_box
        self.T      = T
        self.safe_d = safe_d.float()

        # keep reference formula objects; inputs are supplied later
        # dummy reference traj just to instantiate Expression dims
        ref = torch.zeros((T, 2))
        self.phi_ego = self._build_ego_formula([ref], [ref])   # placeholders
        self.phi_opp = ~self.phi_ego                          # zero-sum

    # ----------------------------------------------------------
    #  Build formulas
    # ----------------------------------------------------------
    def _single_agent_predicates(self, X):
        """Return primitive predicates for **one** ego agent."""
        in_box1, _  = inside_box_kd(X, self.obs_1)
        in_box2, _  = inside_box_kd(X, self.obs_2)
        # goal not used in original 1 v 1 logic, but kept for extension
        reach_goal, _ = inside_box_kd(X, self.goal)

        stay_in_box1 = stlcg.Eventually(
                            stlcg.Always(in_box1, interval=[0, self.T])
                        )
        stay_in_box2 = stlcg.Eventually(
                            stlcg.Always(in_box2, interval=[0, self.T])
                        )
        # reach goal once and hold for 1 step
        reach_goal_once = stlcg.Eventually(
                              stlcg.Always(reach_goal, interval=[0, 1])
                          )

        # outside the circle
        always_out_circle, _ = self._stay_outside_circle(X)

        return stay_in_box1 & stay_in_box2 & reach_goal_once & always_out_circle

    def _stay_outside_circle(self, X):
        d_expr   = stlcg.Expression('d_circ',
                                    torch.norm(X - self.circle[:2]
                                               .unsqueeze(0), dim=-1,
                                               keepdim=True).unsqueeze(0))
        r_expr   = stlcg.Expression('r_circ',
                                    self.circle[2:3]
                                    .unsqueeze(-1).unsqueeze(-1))
        return stlcg.Always(d_expr > r_expr), d_expr

    def _build_ego_formula(self, ego_trajs, opp_trajs):
        """Conjunction over all ego agents incl. pairwise safety."""
        m, n = len(ego_trajs), len(opp_trajs)

        # 1) each ego satisfies reach-avoid on its own
        ego_subs = [self._single_agent_predicates(X) for X in ego_trajs]

        # 2) pairwise safety w.r.t. ALL opponents
        safety_terms = []
        for Xe in ego_trajs:
            for Yo in opp_trajs:
                safe_pred, _ = distance_greater(Xe, Yo, self.safe_d)
                safety_terms.append(stlcg.Always(safe_pred, interval=[0, self.T]))

        phi = ego_subs[0]
        for p in ego_subs[1:] + safety_terms:
            phi = phi & p
        return phi

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------
    def update_formulas(self, ego_trajs: list, opp_trajs: list):
        """
        Call once per rollout before robustness evaluation.
        ego_trajs, opp_trajs : list[Tensor]  each (T,2)
        """
        self.phi_ego = self._build_ego_formula(ego_trajs, opp_trajs)
        self.phi_opp = ~self.phi_ego     # zero-sum

    def compute_robustness_ego(self, ego_trajs, opp_trajs, scale=-1):
        """
        • Build a *temporary* STLFormulaReachAvoid for each ego–opponent set
        (reuse the original helper that already knows its input tree).
        • Team robustness = min over ego agents.
        """
        robvals = []
        for i, Xe in enumerate(ego_trajs):
            # one synthetic opponent trajectory that concatenates ALL opps
            # by taking element-wise min distance (worst case)
            dists = [torch.norm(Xe - Yo, dim=-1, keepdim=True) for Yo in opp_trajs]
            # pick the *closest* opponent at every time step
            closest = torch.min(torch.stack(dists, dim=0), dim=0).values
            # build 1-v-1 helper with same obstacles & goal
            helper = STLFormulaReachAvoidTwoAgents(self.obs_1, self.obs_2,
                                        self.circle, self.circle,  # obs3, obs4 placeholders
                                        self.goal, self.T)
            rob_i = helper.compute_robustness_ego(Xe, closest, scale)
            robvals.append(rob_i)

        return torch.min(torch.stack(robvals))

    def compute_robustness_opp(self, ego_trajs, opp_trajs, scale=-1):
        """
        Opponents maximise *negative* of team robustness.
        """
        return -self.compute_robustness_ego(ego_trajs, opp_trajs, scale)