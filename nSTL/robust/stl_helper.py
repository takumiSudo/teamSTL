from robust.stlcg import stlcg
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def inside_box(xy, obs):
    obs = obs.to(xy.device)
    x = stlcg.Expression('x', xy[:,:1].unsqueeze(0))
    y = stlcg.Expression('y', xy[:,1:].unsqueeze(0))
    r1 = stlcg.Expression('x1', obs[:1].unsqueeze(-1).unsqueeze(-1))
    r2 = stlcg.Expression('x2', obs[1:2].unsqueeze(-1).unsqueeze(-1))
    r3 = stlcg.Expression('y1', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    r4 = stlcg.Expression('y2', obs[3:4].unsqueeze(-1).unsqueeze(-1))
    inputs = ((x,x), (y,y))
    return ((x > r1) & (x < r2)) & ((y > r3) & (y < r4)), inputs
#     ϕ1 = stlcg.Eventually(subformula=stlcg.Always(subformula=inside_box, interval=[0,3]), interval=[0, 12])

def altitude_rule_3d(xyz, zone_bounds, altitude_bounds):
    x = stlcg.Expression('x', xyz[:,:1].unsqueeze(0))
    z = stlcg.Expression('x', xyz[:,2:3].unsqueeze(0))
    zone_low_bound = stlcg.Expression('zone_low', zone_bounds[:1].unsqueeze(-1).unsqueeze(-1))
    zone_upper_bound = stlcg.Expression('zone_upper', zone_bounds[1:2].unsqueeze(-1).unsqueeze(-1))
    altitude_low_bound = stlcg.Expression('low', altitude_bounds[:1].unsqueeze(-1).unsqueeze(-1))
    altitude_upper_bound = stlcg.Expression('upper', altitude_bounds[1:2].unsqueeze(-1).unsqueeze(-1))
    inputs = ((x, x), (z, z))
    return stlcg.Implies(subformula1=((x >= zone_low_bound) & (x <= zone_upper_bound)), subformula2=((z >= altitude_low_bound) & (z <= altitude_upper_bound))), inputs

def always_stay_outside_circle(xy, obs):
    device = xy.device
    center = obs[:2].to(device)
    radius = obs[2:3].to(device)
    d1 = stlcg.Expression('d1',
                          torch.norm(xy - center.unsqueeze(0),
                                     dim=-1, keepdim=True).unsqueeze(0))
    r1 = stlcg.Expression('r', radius.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Always(subformula=(d1 > r1)), d1

def always_stay_outside_unsafe_box_3d(xyz, obs):
    x = stlcg.Expression('x', xyz[:,:1].unsqueeze(0))
    y = stlcg.Expression('y', xyz[:,1:2].unsqueeze(0))
    z = stlcg.Expression('z', xyz[:,2:3].unsqueeze(0))
    r1 = stlcg.Expression('x1', obs[:1].unsqueeze(-1).unsqueeze(-1))
    r2 = stlcg.Expression('x2', obs[1:2].unsqueeze(-1).unsqueeze(-1))
    r3 = stlcg.Expression('y1', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    r4 = stlcg.Expression('y2', obs[3:4].unsqueeze(-1).unsqueeze(-1))
    r5 = stlcg.Expression('z1', obs[4:5].unsqueeze(-1).unsqueeze(-1))
    r6 = stlcg.Expression('z2', obs[5:6].unsqueeze(-1).unsqueeze(-1))
    inputs = (((x,x), (y,y)), (z, z))
    return stlcg.Always(subformula=((((x < r1) | (x > r2)) | ((y < r3) | (y > r4))) | ((z < r5) | (z > r6)))), inputs

def inside_goal_box_3d(xyz, obs):
    x = stlcg.Expression('x', xyz[:,:1].unsqueeze(0))
    y = stlcg.Expression('y', xyz[:,1:2].unsqueeze(0))
    z = stlcg.Expression('z', xyz[:,2:3].unsqueeze(0))
    r1 = stlcg.Expression('x1', obs[:1].unsqueeze(-1).unsqueeze(-1))
    r2 = stlcg.Expression('x2', obs[1:2].unsqueeze(-1).unsqueeze(-1))
    r3 = stlcg.Expression('y1', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    r4 = stlcg.Expression('y2', obs[3:4].unsqueeze(-1).unsqueeze(-1))
    r5 = stlcg.Expression('z1', obs[4:5].unsqueeze(-1).unsqueeze(-1))
    r6 = stlcg.Expression('z2', obs[5:6].unsqueeze(-1).unsqueeze(-1))
    inputs = (((x,x), (y,y)), (z, z))
    return (((x > r1) & (x < r2)) & ((y > r3) & (y < r4))) & ((z > r5) & (z < r6)), inputs

def always_no_collision_between_agents_3d(x1y1z1, x2y2z2, safe_distance=torch.tensor([0.2])):
    d1 = stlcg.Expression('d1', torch.norm(x1y1z1 - x2y2z2.unsqueeze(0), dim=-1, keepdim=True).unsqueeze(0))
    r1 = stlcg.Expression('r', safe_distance.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Always(subformula=(d1 > r1)), d1

def collision_between_agents_3d(x1y1z1, x2y2z2, safe_distance=torch.tensor([0.2])):
    d1 = stlcg.Expression('d1', torch.norm(x1y1z1 - x2y2z2.unsqueeze(0), dim=-1, keepdim=True).unsqueeze(0))
    r1 = stlcg.Expression('r', safe_distance.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Eventually(subformula=(d1 <= r1)), d1

def always_no_collision_between_agents(x1y1, x2y2, safe_distance=torch.tensor([0.2])):
    d1 = stlcg.Expression('d1', torch.norm(x1y1 - x2y2.unsqueeze(0), dim=-1, keepdim=True).unsqueeze(0))
    r1 = stlcg.Expression('r', safe_distance.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Always(subformula=(d1 > r1)), d1

def collision_between_agents(x1y1, x2y2, safe_distance=torch.tensor([0.2])):
    d1 = stlcg.Expression('d1', torch.norm(x1y1 - x2y2.unsqueeze(0), dim=-1, keepdim=True).unsqueeze(0))
    r1 = stlcg.Expression('r', safe_distance.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Eventually(subformula=(d1 <= r1)), d1

def control_limit(u, u_max):
    u_abs = stlcg.Expression('u', u.norm(dim=1, keepdim=True).unsqueeze(0))
    um = stlcg.Expression('umax', u_max.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Always(subformula=(u_abs < um)), u_abs

def velocity_limit(vel_xy, velocity_max):
    vel_xy_abs = stlcg.Expression('u', vel_xy.norm(dim=1, keepdim=True).unsqueeze(0))
    um = stlcg.Expression('umax', velocity_max.unsqueeze(-1).unsqueeze(-1))
    return stlcg.Always(subformula=(vel_xy_abs < um)), vel_xy_abs

class STLFormulaReachAvoid():
    def __init__(self, obs_1, obs_2, obs_3, obs_4, goal, T):

        super().__init__()
        x = torch.arange(-1, 1, 0.04).reshape(-1, 1)
        ref_X = torch.cat((x, x), dim=1)
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        self.obs_3 = obs_3
        self.obs_4 = obs_4
        self.goal = goal
        self.T = T
        self.formula = self.get_formula(ref_X)

    def get_formula(self, X):
        inside_box_1, _ = inside_box(X, self.obs_1)  # ((x,x), (y,y))
        inside_box_2, _ = inside_box(X, self.obs_2)  # ((x,x), (y,y))
        reach_goal, _ = inside_box(X, self.goal)
        
        has_been_inside_box_1 = stlcg.Eventually(subformula=stlcg.Always(subformula=inside_box_1, interval=[0,self.T]))
        has_been_inside_box_2 = stlcg.Eventually(subformula=stlcg.Always(subformula=inside_box_2, interval=[0,self.T]))
        eventually_reach_goal = stlcg.Eventually(subformula=stlcg.Always(subformula=reach_goal, interval=[0,1]))
        always_stay_outside_circle_formula, _ = always_stay_outside_circle(X, self.obs_3)

        formula = (has_been_inside_box_1 & has_been_inside_box_2) & always_stay_outside_circle_formula
        
        return formula
    
    def compute_robustness(self, traj, scale=-1):
        x, y = traj[..., :1], traj[..., 1:]
        box_inputs = ((x, x),(y, y))

        circle_inputs = torch.norm(traj - self.obs_3[:2].unsqueeze(0), dim=-1, keepdim=True)
        robustness = self.formula.robustness(((box_inputs, box_inputs), circle_inputs), scale=scale)
        return robustness
    
    def plot(self, traj, fig_dir='figs/test/', name='test'):
        os.makedirs(fig_dir, exist_ok=True)
        plt.figure(figsize=(10,10))
        # plotting environment
        plt.plot([self.obs_1[0], self.obs_1[0], self.obs_1[1], self.obs_1[1], self.obs_1[0]], [self.obs_1[2], self.obs_1[3], self.obs_1[3], self.obs_1[2], self.obs_1[2]], c="red", linewidth=5)
        plt.plot([self.obs_2[0], self.obs_2[0], self.obs_2[1], self.obs_2[1], self.obs_2[0]], [self.obs_2[2], self.obs_2[3], self.obs_2[3], self.obs_2[2], self.obs_2[2]], c="green", linewidth=5)
        plt.plot([self.obs_4[0], self.obs_4[0], self.obs_4[1], self.obs_4[1], self.obs_4[0]], [self.obs_4[2], self.obs_4[3], self.obs_4[3], self.obs_4[2], self.obs_4[2]], c="orange", linewidth=5)
        plt.plot([self.obs_3[0] + self.obs_3[2].numpy()*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [self.obs_3[1] + self.obs_3[2].numpy()*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c="blue", linewidth=5)

        # plottign optimization results
        plt.plot(traj.detach().numpy()[:,0], traj.detach().numpy()[:,1])
        plt.scatter(traj.detach().numpy()[:,0], traj.detach().numpy()[:,1], c="lightblue", zorder=10)
        plt.scatter([-1,1], [-1,1], s=300)
        plt.axis("equal")
        plt.grid()
        plt.savefig(fig_dir+name+'.png')


class STLFormulaReachAvoidTwoAgents():
    def __init__(self, obs_1, obs_2, obs_3, obs_4, goal, T):

        super().__init__()
        x_ego = torch.arange(-1, 1, 0.04).reshape(-1, 1)
        ref_X_ego = torch.cat((x_ego, x_ego), dim=1)
        x_opp = torch.arange(-1, 1, 0.04).reshape(-1, 1)
        y_opp = torch.arange(1, -1, -0.04).reshape(-1, 1)
        ref_X_opp = torch.cat((x_opp, y_opp), dim=1)
        # ensure reference trajectories live on same device as obstacle tensors
        device = obs_1.device
        ref_X_ego = ref_X_ego.to(device)
        ref_X_opp = ref_X_opp.to(device)
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        self.obs_3 = obs_3
        self.obs_4 = obs_4
        self.goal = goal
        self.T = T
        self.formula_ego = self.get_formula_ego(ref_X_ego, ref_X_opp)
        self.formula_opp = self.get_formula_opp(ref_X_ego, ref_X_opp)

    def get_formula_ego(self, X, Y):
        inside_box_1, _ = inside_box(X, self.obs_1)  # ((x,x), (y,y))
        inside_box_2, _ = inside_box(X, self.obs_2)  # ((x,x), (y,y))
        
        has_been_inside_box_1 = stlcg.Eventually(subformula=stlcg.Always(subformula=inside_box_1, interval=[0,self.T]))
        has_been_inside_box_2 = stlcg.Eventually(subformula=stlcg.Always(subformula=inside_box_2, interval=[0,self.T]))
        always_stay_outside_circle_formula, _ = always_stay_outside_circle(X, self.obs_3)
        always_no_collision_between_agents_formula, _ = always_no_collision_between_agents(X, Y)

        formula = ((has_been_inside_box_1 & has_been_inside_box_2) & always_stay_outside_circle_formula) & always_no_collision_between_agents_formula
        
        return formula

    def get_formula_opp(self, X, Y):
        collision_formula, _ = collision_between_agents(X, Y)

        formula =  collision_formula
        
        return formula
    
    def compute_robustness_ego(self, traj_ego, traj_opp, scale=-1):
        # Extract only the (x,y) coordinates in case traj_ego carries full agent state
        traj_ego_xy = traj_ego[..., :2]
        # Replace traj_ego with only the x-y slice
        traj_ego = traj_ego_xy
        x, y = traj_ego[..., :1], traj_ego[..., 1:]
        box_inputs = ((x, x),(y, y))

        center = self.obs_3[:2]
        if traj_ego.dim() == 3:
            # traj_ego: (B, T, 2)
            center = center.view(1, 1, -1).expand(traj_ego.shape[0], traj_ego.shape[1], -1)
        elif traj_ego.dim() == 2:
            # traj_ego: (T, 2)
            center = center.view(1, -1).expand(traj_ego.shape[0], -1)
        circle_inputs = torch.norm(traj_ego - center, dim=-1, keepdim=True)
        agents_collision_inputs = torch.norm(traj_ego - traj_opp, dim=-1, keepdim=True)
        robustness = self.formula_ego.robustness((((box_inputs, box_inputs), circle_inputs), agents_collision_inputs), scale=scale)
        return robustness
    
    def compute_robustness_opp(self, traj_ego, traj_opp, scale=-1):
        agents_collision_inputs = torch.norm(traj_ego - traj_opp, dim=-1, keepdim=True)
        robustness = self.formula_opp.robustness((agents_collision_inputs), scale=scale)
        return robustness
    
    def animate(self, traj_ego, traj_opp, fig_dir='figs/test/', name='test_two_agents'):
        os.makedirs(fig_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        # plotting environment
        plt.plot([self.obs_1[0], self.obs_1[0], self.obs_1[1], self.obs_1[1], self.obs_1[0]], [self.obs_1[2], self.obs_1[3], self.obs_1[3], self.obs_1[2], self.obs_1[2]], c="red", linewidth=5)
        plt.plot([self.obs_2[0], self.obs_2[0], self.obs_2[1], self.obs_2[1], self.obs_2[0]], [self.obs_2[2], self.obs_2[3], self.obs_2[3], self.obs_2[2], self.obs_2[2]], c="green", linewidth=5)
        plt.plot([self.obs_4[0], self.obs_4[0], self.obs_4[1], self.obs_4[1], self.obs_4[0]], [self.obs_4[2], self.obs_4[3], self.obs_4[3], self.obs_4[2], self.obs_4[2]], c="orange", linewidth=5)
        plt.plot([self.obs_3[0] + self.obs_3[2].numpy()*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [self.obs_3[1] + self.obs_3[2].numpy()*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c="blue", linewidth=5)

        # plottign optimization results
        ego_x, ego_y = traj_ego.detach().numpy()[:,0], traj_ego.detach().numpy()[:,1]
        opp_x, opp_y = traj_opp.detach().numpy()[:,0], traj_opp.detach().numpy()[:,1]
        line_ego, = ax.plot(ego_x, ego_y, c="lightblue", zorder=10)
        line_opp, = ax.plot(opp_x, opp_y, c="red", zorder=10)
        ax.grid()

        ani = animation.FuncAnimation(fig, self.update, ego_x.shape[0], fargs=[ego_x, ego_y, opp_x, opp_y, line_ego, line_opp], interval=100, blit=False)
        ani.save(fig_dir+name+'.gif')
    
    def update(self, num, x, y, opp_x, opp_y, line, line_opp):
        line.set_data(x[:num], y[:num])
        line_opp.set_data(opp_x[:num], opp_y[:num])
        return line, line_opp,


class STLFormulaReachAvoidTwoAgents3D():
    def __init__(self, obs, goal, T, zone1, zone2, altitude1, altitude2):

        super().__init__()
        x_ego = torch.arange(-1, 1, 0.04).reshape(-1, 1)
        ref_X_ego = torch.cat((x_ego, x_ego, x_ego), dim=1)
        x_opp = torch.arange(-1, 1, 0.04).reshape(-1, 1)
        y_opp = torch.arange(1, -1, -0.04).reshape(-1, 1)
        ref_X_opp = torch.cat((x_opp, y_opp, y_opp), dim=1)
        self.obs = obs
        self.goal = goal
        self.T = T
        self.zone1, self.zone2 = zone1, zone2
        self.altitude1, self.altitude2 = altitude1, altitude2
        self.formula_ego = self.get_formula_ego(ref_X_ego, ref_X_opp)
        self.formula_opp = self.get_formula_opp(ref_X_ego, ref_X_opp)

    def get_formula_ego(self, X, Y):
        inside_goal_box, _ = inside_goal_box_3d(X, self.goal)  # ((x,x), (y,y))
        
        has_been_inside_goal_box = stlcg.Eventually(subformula=stlcg.Always(subformula=inside_goal_box, interval=[0,self.T]))
        always_stay_outside_unsafe_box_formula, _ = always_stay_outside_unsafe_box_3d(X, self.obs)
        always_no_collision_between_agents_formula, _ = always_no_collision_between_agents_3d(X, Y)
        altitude_rule1, _ = altitude_rule_3d(X, self.zone1, self.altitude1)
        altitude_rule2, _ = altitude_rule_3d(X, self.zone2, self.altitude2)
        always_altitude_rule1 = stlcg.Always(subformula=altitude_rule1, interval=[0, self.T])
        always_altitude_rule2 = stlcg.Always(subformula=altitude_rule2, interval=[0, self.T])

        formula = ((has_been_inside_goal_box & always_stay_outside_unsafe_box_formula) & (always_altitude_rule1 & always_altitude_rule2)) & always_no_collision_between_agents_formula
        
        return formula

    def get_formula_opp(self, X, Y):
        collision_formula, _ = collision_between_agents_3d(X, Y)

        formula =  collision_formula
        
        return formula
    
    def compute_robustness_ego(self, traj_ego, traj_opp, scale=-1):
        x, y, z = traj_ego[..., :1], traj_ego[..., 1:2], traj_ego[..., 2:3]
        box_inputs = (((x, x),(y, y)),(z,z))
        altitude_inputs = ((x, x), (z, z))
        agents_collision_inputs = torch.norm(traj_ego - traj_opp, dim=-1, keepdim=True)
        robustness = self.formula_ego.robustness((((box_inputs, box_inputs), (altitude_inputs, altitude_inputs)), agents_collision_inputs), scale=scale)
        return robustness
    
    def compute_robustness_opp(self, traj_ego, traj_opp, scale=-1):
        agents_collision_inputs = torch.norm(traj_ego - traj_opp, dim=-1, keepdim=True)
        robustness = self.formula_opp.robustness((agents_collision_inputs), scale=scale)
        return robustness
    
    def animate(self, traj_ego, traj_opp, fig_dir='figs/test/', name='test_two_agents'):
        os.makedirs(fig_dir, exist_ok=True)
        # First initialize the fig variable to a figure
        fig = plt.figure()
        # Add a 3d axis to the figure
        ax = fig.add_subplot(111, projection='3d')
        # plotting environment
        # plt.plot([self.obs[0], self.obs[0], self.obs[1], self.obs[1], self.obs[0], self.obs[0], self.obs[1], self.obs[1]], 
        #          [self.obs[2], self.obs[2], self.obs[2], self.obs[2], self.obs[3], self.obs[3], self.obs[3], self.obs[3]], 
        #          [self.obs[4], self.obs[5], self.obs[4], self.obs[5], self.obs[4], self.obs[5], self.obs[4], self.obs[5]], 
        #          c="red", linewidth=5)

        # Define the vertices of the box
        unsafe_vertices = np.array([
            [self.obs[0], self.obs[2], self.obs[4]],
            [self.obs[0], self.obs[2], self.obs[5]],
            [self.obs[0], self.obs[3], self.obs[4]],
            [self.obs[0], self.obs[3], self.obs[5]],
            [self.obs[1], self.obs[2], self.obs[4]],
            [self.obs[1], self.obs[2], self.obs[5]],
            [self.obs[1], self.obs[3], self.obs[4]],
            [self.obs[1], self.obs[3], self.obs[5]]
        ])

        goal_vertices = np.array([
            [self.goal[0], self.goal[2], self.goal[4]],
            [self.goal[0], self.goal[2], self.goal[5]],
            [self.goal[0], self.goal[3], self.goal[4]],
            [self.goal[0], self.goal[3], self.goal[5]],
            [self.goal[1], self.goal[2], self.goal[4]],
            [self.goal[1], self.goal[2], self.goal[5]],
            [self.goal[1], self.goal[3], self.goal[4]],
            [self.goal[1], self.goal[3], self.goal[5]]
        ])

        # Define the edges of the box
        edges = [
            [0, 1], [0, 2], [1, 3], [2, 3],
            [4, 5], [5, 7], [4, 6], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # Plot the edges of the unsafe box and goal box
        for edge in edges:
            x = [unsafe_vertices[edge[0]][0], unsafe_vertices[edge[1]][0]]
            y = [unsafe_vertices[edge[0]][1], unsafe_vertices[edge[1]][1]]
            z = [unsafe_vertices[edge[0]][2], unsafe_vertices[edge[1]][2]]
            ax.plot(x, y, z, color='red')

            x = [goal_vertices[edge[0]][0], goal_vertices[edge[1]][0]]
            y = [goal_vertices[edge[0]][1], goal_vertices[edge[1]][1]]
            z = [goal_vertices[edge[0]][2], goal_vertices[edge[1]][2]]
            ax.plot(x, y, z, color='green')

        # set axis range
        ax.set_xlim([-3, 3])
        ax.set_ylim([-1, 3])
        ax.set_zlim([0, 2])


        # plottign optimization results
        ego_x, ego_y, ego_z = traj_ego.detach().numpy()[:,0], traj_ego.detach().numpy()[:,1], traj_ego.detach().numpy()[:,2]
        opp_x, opp_y, opp_z = traj_opp.detach().numpy()[:,0], traj_opp.detach().numpy()[:,1], traj_opp.detach().numpy()[:,2]
        line_ego, = ax.plot(ego_x, ego_y, ego_z, c="lightblue", zorder=10)
        line_opp, = ax.plot(opp_x, opp_y, opp_z, c="black", zorder=10)
        ax.grid()

        ani = animation.FuncAnimation(fig, self.update, ego_x.shape[0], fargs=[ego_x, ego_y, ego_z, opp_x, opp_y, opp_z, line_ego, line_opp], interval=100, blit=False)
        ani.save(fig_dir+name+'.gif')
    
    def update(self, num, x, y, z, opp_x, opp_y, opp_z, line, line_opp):
        line.set_data(x[:num], y[:num])
        line.set_3d_properties(z[:num])
        line_opp.set_data(opp_x[:num], opp_y[:num])
        line_opp.set_3d_properties(opp_z[:num])
        return line, line_opp,