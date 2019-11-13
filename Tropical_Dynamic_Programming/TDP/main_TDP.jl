function main_TDP(itermax, epsilon, x0, T, Xi, dom, dyn, cost)
"""

Solves a risk-neutral linear-convexPolyhedral discrete time dynamical system
with noises of finite support.

Output description:

Returns sets of basic functions building upper approximations of the
value functions of a discrete time  through
a Min-Plus algorithm and lower approximations through the SDDP algorithm.
Also return the optimal policy for the last lower approximations.
    Could easily be modified for other upper and lower approximations scheme.

Input description:

itermax, maximal number of iterations of TDP.

epsilon, stopping criterium: if the gap between upper and lower approximations
is less than epsilon, then stop.

x0, initial state of the dynamical system.

T, finite horizon of the dynamical system.

Xi, is an array of length T-1 containing at each component a vector (of length
that may vary on the time) describing the randomness occuring at the time t,
which is assumed to be of finite support and independant of the past.

dyn, is an array of length T-1 containing at each component a triplet of
matrices A_t, B_t, C_t. They caracterize the linear dynamic of the dynamical
system:
x_(t+1) = A_t*x_t + B_t*u_t + C_t*w_t,
where x_t stands for the state at time t,
    u_t stands for the control applied to x_t,
        w_t stands for the randomness observed before applying the control u_t.
(Hazard-decision framework)

cost, is an array of length T which contains T-1 matrices P_t, T-1
vectors b_t and T array of size I, (c_t^i), i in 1:I. It caracterizes a
polyhedral function for each t:
c(t, x_t, u_t, w_t) is polyhedral in (x_t, u_t, w_t), i.e. for a given time t,
(x_t, u_t, w_t) belongs to some polyhedron caracterized by a matrix P_t and
a vector b_t:
    P_t * (x_t, u_t, w_t)^T <= b_t
and
for some vector c_t the instantaneous cost at (x_t, u_t, w_t) is:
    max_{i in 1:I} < c_t^i, (x_t, u_t, w_t) >
"""

gap_at0 = Inf;
iter = 0;

# Initialization (to finish)
F_up = Array{Any}(undef, T);
F_down = Array{Any}(undef, T);
PC_traj = ones(x_0, T); # Assuming that x_0 is admissible for all time

# Iteration step
while (gap_at0 > epsilon) && (iter < itermax)
    global F_up, F_down, gap_at0, best_u, iter;

    # Forward phase: generates the Problem-Child trajectory
    PC_traj = Problem_Child_trajectory(F_up, F_down, x0, best_u_down,
            dyn, T, Xi);

    # Backward phase: improve the current approximations, stocks best_u and val
    new_up = Selection_MinPlus(PC_traj, F_up, dom, dyn, cost);
    new_down = Selection_SDDP(PC_traj, F_down, dom, dyn, cost);

    F_up = hcat(F_up, new_up[1]);
    F_down = hcat(F_down, new_down[1]);

    best_u_up = new_up[2];
    best_u_down = new_down[2];

    val_up = new_up[3];
    val_down = new_down[3];

    # Stopping criterion update
    gap_at0 = val_up[1] - val_down[1];

    # Saving data

    iter += 1;
end
return(F_up, F_down, best_u)
end
