#Eitan Chelly 346395064, Onn Azulay 325759173
#potential Differential Game setup (relative work paper)
using LinearAlgebra, Random, PyPlot, StatsBase

# === TRACK CONSTANTS ===
const R_inner = 30.0
const track_width = 12.0
const R_outer = R_inner + track_width
const light_width = 3.0

const N_centerline = 200
const R_center = (R_outer + R_inner)/2
centerline = [ [R_center * cos(θ), R_center * sin(θ)] for θ in range(0, 2π, length=N_centerline+1) ]
inner_radius = [ [R_inner * cos(θ), R_inner * sin(θ)] for θ in range(0, 2π, length=N_centerline+1) ]

# === LIGHT ZONES ===
const light_zones = [
    (center=0.0,     half_angle=0.3, side="inner"),
    (center=π / 4,     half_angle=0.3, side="inner"),
    (center=2π / 3,  half_angle=0.3, side="outer"),
    (center=4π / 3,  half_angle=0.3, side="inner")
]

# === DESIRED SPEEDS ===
const v_des_blue = 8.0
const v_des_red  = 10.0

# === CONTROL GAINS (SEPARATE FOR EACH AGENT) ===
const Kp_v_blue = 0.5
const Kp_v_red  = 0.5
const Kp_r_blue = 0.5
const Kp_r_red  = 1.0

# === PHYSICAL LIMITS / PENALTIES ===
const dt = 0.05
const max_time = 20.0
const a_max = 9.0
const λ_collision = 0.2 #lower for less collisions
const k_collision = 5.0
const ε = 1e-4        # Small value to prevent division by zero in gradient normalization

const w_v = 2.0
const w_c = 10.0
const w_T = 10.0
const w_r = 12.0 



# === AGENT STRUCTURES ===
mutable struct AgentState
    x::Float64 #horizontal location grid
    y::Float64 #vertical location grid
    vx::Float64 #horizontal velocity
    vy::Float64 #vertical velocity
    s::Float64   # distance traveled along track
end

mutable struct AgentBelief #Belief structure
    mean::Vector{Float64}
    cov::Matrix{Float64}
end

struct CarModel
    c_drag::Float64    # Longitudinal drag coefficient
    c_slip::Float64    # Lateral slip coefficient
end


mutable struct CognitiveLog
    t::Vector{Float64} #Stores the time stamps (in seconds) for each step of the simulation
    blue_ar::Vector{Float64} #radial accelaration blue
    red_ar::Vector{Float64} #radial accelaration red
    delta_r_est::Vector{Float64} #the radial estimation error from blue’s perspective
    delta_s::Vector{Float64} #the longitudinal distance error between red and blue
    delta_r_est_red::Vector{Float64} #the radial estimation error from red’s perspective
    collision_penalty_blue::Vector{Float64} #the collision penalty computed from blue’s belief,
    #based on Mahalanobis distance between the two agents’ position beliefs.
    collision_penalty_red::Vector{Float64} #the collision penalty computed from red’s belief,
    #based on Mahalanobis distance between the two agents’ position beliefs.
end

# === LIGHT ZONE DETECTION ===
# Function: in_light_zone
# ------------------------
# Inputs:
#   x, y         - Cartesian coordinates of agent's position
#   light_zones  - Array of light zone specifications (angle + radius bounds)
# Output:
#   Returns true if the position is within any light zone; otherwise false
# Purpose:
#   Determines if an agent's position falls inside any of the defined light zones,
#   which affect observation quality. These zones are defined in angular sections
#   on either the inner or outer radius of the track.
function in_light_zone(x::Float64, y::Float64, light_zones)
    #radial coordinates definition
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ #Maps angle to range [0,2π) instead of (−π,π]
    for zone in light_zones
        #computes the smallest angular difference between θθ and the center of the light zone
        dθ = abs(mod(θ - zone.center + π, 2π) - π)
        if dθ <= zone.half_angle
            if zone.side == "inner" && r >= R_inner && r <= R_inner + light_width
                return true
            elseif zone.side == "outer" && r <= R_outer && r >= R_outer - light_width
                return true
            end
        end
    end
    return false
end


# === UNIT TANGENT & RADIAL VECTORS ===
# Function: unit_vectors
# -----------------------
# Inputs:
#   x, y - Agent position in Cartesian coordinates
# Output:
#   t_hat - Unit tangent vector at position
#   r_hat - Unit radial vector at position
#   θ     - Angle of position vector from origin
# Purpose:
#   Computes the local tangential and radial unit vectors from Cartesian position.
#   These are used to convert between global and track-relative directions.
function unit_vectors(x::Float64, y::Float64)
    θ = atan(y, x)
    t_hat = [-sin(θ), cos(θ)]  # Tangential
    r_hat = [cos(θ),  sin(θ)]  # Radial
    return t_hat, r_hat, θ
end

# === PROPAGATE TRUE STATE ===
# Function: propagate_state!
# ---------------------------
# Inputs:
#   agent - AgentState struct (mutable)
#   a_t   - Tangential acceleration input
#   a_r   - Radial acceleration input
#   dt    - Time step
# Output:
#   Updates the agent's state in-place
# Purpose:
#   Integrates the agent's true dynamics forward in time, applying physical constraints
#   such as boundary collisions and enforcing no backward motion along the track.
function propagate_state!(agent::AgentState, a_t::Float64, a_r::Float64, dt::Float64)
    t_hat, r_hat, _ = unit_vectors(agent.x, agent.y) #Compute Local Axes 
    #Compute Local Acceleration in directions
    ax = a_t * t_hat[1] + a_r * r_hat[1]
    ay = a_t * t_hat[2] + a_r * r_hat[2]

    #standard kinamatics based constant speed (true locally)
    agent.vx += ax * dt
    agent.vy += ay * dt
    agent.x  += agent.vx * dt
    agent.y  += agent.vy * dt
    agent.s  += sqrt(agent.vx^2 + agent.vy^2) * dt #Compute arc-length increment

    # Track boundary collision response
    r = sqrt(agent.x^2 + agent.y^2)
    nx, ny = agent.x / r, agent.y / r #normal vector n pointing outward
    if r < R_inner
        agent.x, agent.y = nx * R_inner, ny * R_inner #Snap the position back to exactly on the inner boundary.
        vr = agent.vx * nx + agent.vy * ny
        if vr < 0
            #If the agent is moving into the wall, Reflect the radial velocity like an elastic bounce.
            agent.vx -= 2 * vr * nx
            agent.vy -= 2 * vr * ny
        end
    elseif r > R_outer
        agent.x, agent.y = nx * R_outer, ny * R_outer #Snap to the outer boundary
        vr = agent.vx * nx + agent.vy * ny
        if vr > 0
            agent.vx -= 2 * vr * nx
            agent.vy -= 2 * vr * ny
        end
    end

    # Prevent backward motion
    t_hat_new, _, _ = unit_vectors(agent.x, agent.y)
    v_t = agent.vx * t_hat_new[1] + agent.vy * t_hat_new[2] #Compute tangential velocity component
    if v_t < 0
        #If agent is going backward along the track, 
        #Remove tangential component so velocity has zero projection onto track direction
        agent.vx -= v_t * t_hat_new[1]
        agent.vy -= v_t * t_hat_new[2]
    end
end

# === CREATE INITIAL BELIEF ===
# Function: create_belief
# ------------------------
# Inputs:
#   x, y, vx, vy - Initial state estimates
#   std_devs     - Vector of standard deviations for belief uncertainty
# Output:
#   Returns a new AgentBelief struct with specified mean and covariance
# Purpose:
#   Initializes a Gaussian belief over the agent's state based on initial
#   estimates and uncertainty levels.
function create_belief(x, y, vx, vy, std_devs)
    mean = Float64[x, y, vx, vy]
    cov = Diagonal(std_devs .^ 2)
    return AgentBelief(mean, cov)
end


# === BELIEF PREDICTION (MOTION MODEL + COVARIANCE PROP) ===
# Function : belief_predict!
#
# Purpose:
# This function performs the prediction step of an EKF
# for an agent. It updates the agent's belief
# state (both mean and covariance) using a second-order motion model that
# includes drag and lateral slip effects. The prediction assumes constant
# acceleration over the time step (locally).
#
# Inputs:
# - belief::AgentBelief
#     A mutable structure holding the agent's current belief, consisting of:
#     - mean::Vector{Float64} → [x, y, vx, vy] (position and velocity)
#     - cov::Matrix{Float64} → 4×4 state covariance matrix
# - a_t::Float64
#     Tangential acceleration command (aligned with the direction of motion).
# - a_r::Float64
#     Radial acceleration command (perpendicular to direction of motion).
# - dt::Float64
#     Time step duration (in seconds).
# - model::CarModel
#     Struct containing physical parameters of the vehicle:
#     - c_drag: longitudinal aerodynamic drag coefficient
#     - c_slip: lateral velocity damping coefficient ("slip")
# Outputs:
# - This is an in-place update. The belief.mean and belief.cov are both updated:
#     - belief.mean is advanced using Euler integration with drag/slip modeling.
#     - belief.cov is updated using the linearized Jacobian A and process noise Q.
# Notes:
# - The drag and slip effects are modeled as speed-proportional forces.
# - The process noise is dynamically scaled with velocity, capturing higher
#   uncertainty at higher speeds.
# - The Jacobian A captures the partial derivatives of the motion model and
#   allows linear propagation of uncertainty.
# =============================================================================

function belief_predict!(belief::AgentBelief, a_t, a_r, dt, model::CarModel)
    #the current estimate of the agent's position and velocity
    μ = belief.mean
    x, y, vx, vy = μ #assigning from AgentState struct

    # === Compute unit vectors based on current position ===
    t_hat, r_hat, _ = unit_vectors(x, y)

    # === Current speed and direction ===
    v = sqrt(vx^2 + vy^2)

    # === Apply drag and slip (speed proportional) ===
    a_drag = -model.c_drag * v           # Longitudinal drag
    a_slip = -model.c_slip * v           # Lateral slip

    # === Tangential and radial accelerations in global frame ===
    #Global-frame decomposition of total acceleration
    ax = (a_t + a_drag) * t_hat[1] + (a_r + a_slip) * r_hat[1]
    ay = (a_t + a_drag) * t_hat[2] + (a_r + a_slip) * r_hat[2]

    #implementing the motion model (kinematic law, under constant accelarations)
    #or Euler integration for motion... same thing
    μ[1] = x + vx * dt
    μ[2] = y + vy * dt
    μ[3] = vx + ax * dt
    μ[4] = vy + ay * dt
    belief.mean = μ

    #Constructs the Jacobian matrix A of the motion model 
    A = Matrix{Float64}(I, 4, 4)
    A[1, 3] = dt #position depends linearly on velocity, assigning it directly
    A[2, 4] = dt #position depends linearly on velocity, assigning it directly

    base_noise = 0.2 #noise modeled as just remaining in the existing velocity
    gain_noise = 0.05 #gained noise cause the agents accelarate...
    qv = (base_noise + gain_noise * v)^2
    #process noise covariance matrix    
    Q = Diagonal([0.0, 0.0, qv, qv])

    belief.cov = A * belief.cov * A' + Q #EKF covariance update
end

#non-mutating version, Needed for simulating belief rollouts in optimizer.
#A full deep copy is made so that we can simulate the effect of the control
# inputs on a new belief, leaving the original untouched
function propagate_belief(belief::AgentBelief, a_t::Float64, a_r::Float64, dt::Float64, model::CarModel)
    new_belief = deepcopy(belief)
    belief_predict!(new_belief, a_t, a_r, dt, model)
    return new_belief
end


# === BELIEF UPDATE (OBSERVATION MODEL) ===
# Function: belief_update!
# -------------------------
# Inputs:
#   belief - Agent's prior belief (mutable)
#   z      - Measurement vector [x, y]
#   R      - Measurement noise covariance matrix
# Output:
#   Updates belief.mean and belief.cov in-place
# Purpose:
#   Performs the update step of the EKF to incorporate
#   a new position measurement. 
function belief_update!(belief::AgentBelief, z::Vector{Float64}, R::AbstractMatrix{<:Float64})
    #extracts the first two components of the full state vector [x,y,vx,vy]
    H = [1.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0] #geometry setup, observation matrix... 
    S = H * belief.cov * H' + R #innovation covariance
    K = belief.cov * H' * inv(S) #Kalman Gain (optimal gain)
    y_tilde = z .- H * belief.mean #the observation estimation residual
    belief.mean += K * y_tilde #Filtering the state expectation
    
    #Joseph's formula for Filtering the Covariance error
    I4 = Matrix{Float64}(I, 4, 4)
    KH = K * H
    belief.cov = (I4 - KH) * belief.cov * (I4 - KH)' + K * R * K'

end


# === UTILITY: PROJECT POSITION TO TRACK ARC-LENGTH ===
# Function: project_to_track
# ---------------------------
# Inputs:
#   centerline - Vector of 2D points defining the track centerline
#   pos        - 2D position vector of the agent [x, y]
# Output:
#   s_progress - Arc-length distance along the centerline to the
#                closest projected point from the input position
# Purpose:
#   Computes the closest projection of the input position onto the
#   piecewise-linear centerline path, and returns the accumulated
#   arc-length up to that projection point. We would like to get some feeling
#on the tangential progress made along the track
function project_to_track(centerline::Vector{Vector{Float64}}, pos::Vector{Float64})
    min_dist = Inf                     # minimal distance initialized
    s_progress = 0.0                   # projected arc-length progress
    total = 0.0                        # cumulative arc-length along centerline

    for i in 1:length(centerline)-1
        p1, p2 = centerline[i], centerline[i+1]      # segment endpoints
        seg = p2 - p1                                # segment vector
        seg_len = norm(seg)                          # segment length
        if seg_len < 1e-6 #Skip degenerate segments (e.g., duplicated points) to avoid numerical issues
            continue
        end

        #Project pos onto the line defined by the segment and then clamp to stay within the segment
        t = clamp(dot(pos - p1, seg) / (seg_len^2), 0.0, 1.0)
        #Compute the actual projected point
        proj = p1 + t * seg
        #Euclidean distance from pos to proj
        dist = norm(pos - proj)

        # check if this segment gives the closest projection
        if dist < min_dist
            min_dist = dist
            s_progress = total + t * seg_len
        end

        total += seg_len
    end

    return s_progress
end


# === TRACK DEVIATION: SIGNED LATERAL OFFSET FROM CENTERLINE ===
# Function: signed_lateral_error
# ------------------------------
# Inputs:
#   pos        - 2D position vector of the agent [x, y]
#   centerline - Vector of 2D points defining the track centerline
# Output:
#   signed_err - Scalar value representing the signed lateral deviation
#                from the centerline; positive means outward (to the left),
#                negative means inward (to the right), relative to track direction
# Purpose:
#   Computes the minimum-distance projection of the agent position onto
#   the track centerline and returns the **signed** lateral error using
#   the track-normal vector. This allows determining whether the agent
#   is to the left or right of the track path.
# **The lateral error is the shortest distance from the point to the centerline,
#   where the sign indicates whether the point is to the left (positive) or right
#   (negative) of the local centerline direction.
function signed_lateral_error(pos::Vector{Float64}, centerline::Vector{Vector{Float64}})
    min_dist = Inf            # Initialize minimum Euclidean distance
    signed_err = 0.0          # Initialize signed error result

    for i in 1:length(centerline)-1
        p1, p2 = centerline[i], centerline[i+1]    # segment endpoints
        seg = p2 - p1                              # segment vector
        seg_len = norm(seg)                        # segment length
        if seg_len < 1e-6 #Skip degenerate segments (e.g., duplicated points) to avoid numerical issues
            continue
        end

        # project pos onto segment
        t = clamp(dot(pos - p1, seg) / (seg_len^2), 0.0, 1.0)
        proj = p1 + t * seg
        dist = norm(pos - proj)

        # update closest projection and compute signed error
        if dist < min_dist
            min_dist = dist
            track_dir = seg / seg_len #Normalize the segment direction
            track_normal = [-track_dir[2], track_dir[1]]  # 90° CCW rotation
            signed_err = dot(pos - proj, track_normal) #Project the vector from projection to pos onto the normal
        end
    end

    return signed_err
end

# === SYMMETRIC CONTROLLER FOR BELIEF-BASED TRACKING AND COLLISION AVOIDANCE ===
# Function: compute_control_symmetric
# -----------------------------------
# Inputs:
#   self_belief     - AgentBelief object representing the agent's own belief
#   belief_of_other - AgentBelief object representing the agent's belief about the opponent
#   v_des           - Desired forward (tangential) velocity
#   Kp_v            - Proportional gain for tangential speed control
#   Kp_r            - Proportional gain for radial (track-centering) control
#
# Output:
#   a_t             - Tangential acceleration to maintain desired speed and avoid collision
#   a_r             - Radial acceleration to steer the agent towards the desired track radius
#
# Purpose:
#   This controller computes tangential and radial accelerations for an agent based on its
#   estimated state and belief about the opponent. It performs:
#     • Speed regulation using proportional control
#     • Collision avoidance based on Mahalanobis distance in position uncertainty
#     • Radial correction to steer the agent toward a nominal track radius
#
#   The collision penalty uses a Gaussian-shaped repulsion based on belief covariance.
#   The radial correction guides the agent to follow the centerline via lateral projection.
function compute_control_symmetric(
    self_belief::AgentBelief, # The agent’s current state estimate and covariance
    belief_of_other::AgentBelief, #The belief of the other agent (used for collision avoidance)
    v_des::Float64, #Desired speed (target tangential velocity)
    Kp_v::Float64, #Proportional gain for tangential velocity tracking
    Kp_r::Float64, #Proportional gain for radial deviation correction
)
    #loading the zero and first order beliefs
    μ_self = self_belief.mean
    μ_other = belief_of_other.mean
    Σ_self = self_belief.cov[1:2, 1:2]
    Σ_other = belief_of_other.cov[1:2, 1:2]

    # === Collision avoidance ===
    Δμ = μ_other[1:2] - μ_self[1:2]                  # relative position
    Σ_sum = Σ_self + Σ_other + 1e-6I                 # combined uncertainty (+ 1e-6I to avoid singularity...)
    D_c_squared = dot(Δμ, Σ_sum \ Δμ)                # Mahalanobis distance squared
    penalty = exp(-0.5 * D_c_squared / λ_collision)  #soft penalty function, shaped like a Gaussian
    #Small when agents are close (large penalty), Near 1 when agents are far apart

    # === Tangential acceleration ===
    v_vec = μ_self[3:4]
    v_tan = norm(v_vec)                              # current tangential velocity
    a_t_base = Kp_v * (v_des - v_tan)                # proportional speed controller
    a_t = clamp(a_t_base - k_collision * penalty, -a_max, a_max) #Clamp result to limits [−a_max,a_max]

    # === Radial acceleration towards centerline ===
    r_current = norm(self_belief.mean[1:2])
    r_target = R_inner + 0.5                          # (+0.5 to avoid collision into track boundary)
    a_r_base = -Kp_r * (r_current - r_target)         #PD control to drive agent back to r=rtargetr=rtarget​
    a_r = clamp(a_r_base, -a_max, a_max)

    return a_t, a_r
end


# === TRACK LENGTH COMPUTATION ===
# Function: compute_s_goal
# -------------------------
# Inputs:
#   centerline - Vector of 2D points representing the discretized track centerline
#
# Output:
#   total      - Scalar value representing the total arc-length of the closed track
#
# Purpose:
#   Computes the total arc-length (curvilinear length) of a closed-loop track by summing
#   the Euclidean distances between consecutive points on the discretized centerline.
#   Assumes that the centerline forms a continuous loop (i.e., wraps around).
#
#   The arc-length result is used as a goal parameter (s_goal) for agent planning.
function compute_s_goal(centerline)
    total = 0.0 #accumulator for arc-length
    N = length(centerline) #N: number of discrete points (segments) along the centerline
    for i in 1:N
        p1 = centerline[i] #current point
        p2 = centerline[mod1(i + 1, N)]  #next point in sequence (mod1(i+1, N) wraps around
        # so that after the last point, you go back to the first point).
        total += norm(p2 - p1)
    end
    return total
end

s_goal = compute_s_goal(centerline)  # total length of the closed track


# === CLOSEST POINT ON INNER TRACK BOUNDARY ===
# Function: closest_point_on_inner_radius
# ---------------------------------------
# Inputs:
#   inner_radius - Vector of 2D points defining the discretized inner boundary of the track
#   pos          - 2D position vector [x, y] of the agent
#
# Output:
#   closest      - 2D point on the inner_radius that is closest to the given position
#
# Purpose:
#   Identifies the point on the inner track boundary that is closest to a given position.
#   This is used in scenarios where proximity to the track wall is important,
#   e.g., for visualization, diagnostics, or safety constraints in planning or control.
function closest_point_on_inner_radius(inner_radius::Vector{Vector{Float64}}, pos::Vector{Float64})
    min_dist = Inf
    closest = inner_radius[1] #initialized to the first point in the inner_radius array
    for pt in inner_radius #For each point pt on the inner radius
        d = norm(pos - pt) #Compute Euclidean distance
        if d < min_dist
            min_dist = d
            closest = pt
        end
    end
    return closest
end

# === CONTROL GRADIENT COMPUTATION FOR OPTIMIZATION ===
# Function: compute_control_gradient
# ----------------------------------
# Inputs:
#   beliefs     - Vector of self-belief states over the planning horizon
#   controls    - Vector of control inputs (a_t, a_r) over the horizon
#   b_opponent  - Vector of opponent belief states over the horizon
#   λ_speed     - Gain for speed tracking penalty
#   v_des       - Desired target speed
#   R_inner     - Inner track radius (for radial deviation penalty)
#   dt          - Time step
#   s_goal      - Desired arc-length progress target
#   w_T         - Weight for terminal arc-length progress cost
#   w_r         - Weight for radial deviation cost
#
# Output:
#   grads       - Vector of gradients ∂J/∂(a_t, a_r) for each time step,
#                 where J is the cost function to minimize
#
# Purpose:
#   Computes the gradient of the agent's total cost function with respect to its control inputs
#   over a planning horizon. The cost includes:
#     • Speed tracking error (c_track)
#     • Radial position deviation from the centerline
#     • Collision avoidance with opponent (c_coll)
#     • Final arc-length progress at the end of the horizon (r(p))
#   These gradients are used in gradient-descent-based optimization schemes (e.g., potential games).
function compute_control_gradient(
    beliefs::Vector{AgentBelief},
    controls::Vector{Tuple{Float64, Float64}},
    b_opponent::Vector{AgentBelief},
    λ_speed::Float64,
    v_des::Float64,
    R_inner::Float64,
    dt::Float64,
    s_goal::Float64,
    w_T::Float64,
    w_r::Float64
)::Vector{Tuple{Float64, Float64}}

    H = length(controls) # the prediction horizon length
    grads = Vector{Tuple{Float64, Float64}}(undef, H) #stores gradients of
    # the cost at each time step w.r.t. tangential and radial acceleration.

    for t in 1:H #at each time of the horizon
        b = beliefs[t]
        μ = b.mean
        vx, vy = μ[3], μ[4] 
        speed = sqrt(vx^2 + vy^2)

        # === Speed tracking gradient (J_speed = 0.5*λ_speed*(v_des - speed)^2, Quadratic penalty) ===
        dJ_da_t = -λ_speed * (v_des - speed) #The cost for not matching desired speed

        # === Radial deviation gradient  (J_radial = w_r*(r_mag - R_inner)^2, Quadratic penalty)===
        r_pos = μ[1:2]
        r_mag = norm(r_pos)
        r_err = r_mag - R_inner
        dr_da_r = dt^2  # Assumes a_r directly affects radial distance quadratically over time
        dJ_da_r = 2 * w_r * r_err * dr_da_r

        # === Collision penalty gradient (J_coll ∝ exp(-0.5 * D_c_squared / λ_collision)) ===
        Δμ = b_opponent[t].mean[1:2] - r_pos
        Σ_sum = b.cov[1:2, 1:2] + b_opponent[t].cov[1:2, 1:2] + 1e-6I
        D_c_squared = dot(Δμ, Σ_sum \ Δμ)
        penalty = exp(-0.5 * D_c_squared / λ_collision) #assuming the soft penalty for collision is 
        #proportinal to the gaussian
        dJ_da_t += k_collision * penalty

        # === Terminal arc-length progress gradient (only at final time) ===
        # (J_goal = w_T*(s_T - s_goal)^2, Quadratic penalty)
        if t == H
            s_T = project_to_track(centerline, r_pos) #At final step, project current position to track
            ds_da_t = dt
            dJ_da_t += 2 * w_T * (s_T - s_goal) * ds_da_t
        end

        grads[t] = (dJ_da_t * dt, dJ_da_r * dt)
    end

    return grads
end



function simulate_forward_dynamics_potential(
    self_belief::AgentBelief, opponent_belief::AgentBelief,
    self_controls::Vector{Tuple{Float64, Float64}},
    opponent_controls::Vector{Tuple{Float64, Float64}},
    horizon::Int,
    v_des_self::Float64, v_des_oppo::Float64,
    R_inner::Float64, s_goal::Float64,
    w_v::Float64, w_r::Float64, w_c::Float64, w_T::Float64,
    model_self::CarModel, model_oppo::CarModel
)::Tuple{Vector{AgentBelief}, Vector{AgentBelief}, Float64}

    b_self = deepcopy(self_belief)
    b_oppo = deepcopy(opponent_belief)

    beliefs_self = AgentBelief[]
    beliefs_oppo = AgentBelief[]
    total_cost = 0.0

    for t in 1:horizon
        a_t_self = self_controls[t]
        a_t_oppo = opponent_controls[t]

        belief_predict!(b_self, a_t_self[1], a_t_self[2], dt, model_self)
        belief_predict!(b_oppo, a_t_oppo[1], a_t_oppo[2], dt, model_oppo)

        push!(beliefs_self, deepcopy(b_self))
        push!(beliefs_oppo, deepcopy(b_oppo))

        # === SPEED COST ===
        v_self = norm(b_self.mean[3:4])
        v_oppo = norm(b_oppo.mean[3:4])
        cost_v_self = w_v * (v_self - v_des_self)^2
        cost_v_oppo = w_v * (v_oppo - v_des_oppo)^2

        # === RADIAL DEVIATION COST ===
        r_self = norm(b_self.mean[1:2])
        r_oppo = norm(b_oppo.mean[1:2])
        cost_r_self = w_r * (r_self - R_inner)^2
        cost_r_oppo = w_r * (r_oppo - R_inner)^2

        # === COLLISION COST (Mahalanobis distance) ===
        Δμ = b_oppo.mean[1:2] - b_self.mean[1:2]
        Σ_sum = b_self.cov[1:2, 1:2] + b_oppo.cov[1:2, 1:2] + 1e-6I
        D_M_sq = dot(Δμ, Σ_sum \ Δμ)
        cost_collision = w_c * exp(-0.5 * D_M_sq / λ_collision)

        # === ACCUMULATE COST (tracking and anti-collision costs that build the potential) ===
        total_cost += cost_v_self + cost_v_oppo + cost_r_self + cost_r_oppo + cost_collision
    end

    # === TERMINAL ARC-LENGTH COSTS ===
    μ_final_self = b_self.mean[1:2]
    μ_final_oppo = b_oppo.mean[1:2]
    s_T_self = project_to_track(centerline, μ_final_self)
    s_T_oppo = project_to_track(centerline, μ_final_oppo)
    cost_T_self = w_T * (s_goal - s_T_self)^2
    cost_T_oppo = w_T * (s_goal - s_T_oppo)^2

    total_cost += cost_T_self + cost_T_oppo

    return beliefs_self, beliefs_oppo, total_cost
end



function update_controls_with_gradient_potential(
    controls::Vector{Tuple{Float64, Float64}},
    gradients::Vector{Tuple{Float64, Float64}},
    η::Float64,
    ε::Float64
)::Vector{Tuple{Float64, Float64}}

    H = length(controls)
    updated = Vector{Tuple{Float64, Float64}}(undef, H)

    for t in 1:H
        a_t, a_r = controls[t]
        g_t, g_r = gradients[t]

        # Regularized normalized gradient descent step
        new_a_t = clamp(a_t - η * g_t / (abs(g_t) + ε), -a_max, a_max)
        new_a_r = clamp(a_r - η * g_r / (abs(g_r) + ε), -a_max, a_max)

        updated[t] = (new_a_t, new_a_r)
    end

    return updated
end

# === NASH EQUILIBRIUM CONTROL VIA DIFFERENTIAL POTENTIAL GAME SETUP ===
function compute_nash_controls!(
    blue_belief_self::AgentBelief, red_belief_self::AgentBelief,
    blue_belief_red::AgentBelief, red_belief_blue::AgentBelief,
    v_des_blue::Float64, v_des_red::Float64,
    Kp_v_blue::Float64, Kp_v_red::Float64;
    model_blue::CarModel, model_red::CarModel,
    H::Int = 10,
    η::Float64 = 0.1,
    n_iter::Int = 20,
    ε::Float64 = 1e-3
)

    # === INITIALIZE CONTROL SEQUENCES ===
    blue_controls = [(0.0, 0.0) for _ in 1:H]
    red_controls  = [(0.0, 0.0) for _ in 1:H]

    # Optional heuristic initialization
    for t in 1:H
        blue_controls[t] = compute_control_symmetric(
            blue_belief_self, blue_belief_red, v_des_blue, Kp_v_blue, Kp_r_blue
        )
        red_controls[t] = compute_control_symmetric(
            red_belief_self, red_belief_blue, v_des_red, Kp_v_red, Kp_r_red
        )
    end

    for _ in 1:n_iter
        # === FORWARD PASS: propagate beliefs and compute total potential cost ===
        blue_beliefs, red_beliefs, _ = simulate_forward_dynamics_potential(
        blue_belief_self, red_belief_self,
        blue_controls, red_controls,
        H, v_des_blue, v_des_red, R_inner, s_goal,
        w_v, w_c, w_T, w_r,
        model_blue, model_red
)


        # === BACKWARD PASS: compute gradients of potential function ===
        grad_blue = compute_control_gradient(
            blue_beliefs, blue_controls,
            red_beliefs,
            Kp_v_blue, v_des_blue, R_inner, dt,
            s_goal, w_T, w_r
        )

        grad_red = compute_control_gradient(
            red_beliefs, red_controls,
            blue_beliefs,
            Kp_v_red, v_des_red, R_inner, dt,
            s_goal, w_T, w_r
        )

        # === GRADIENT-BASED CONTROL UPDATE (normalized) ===
        blue_controls = update_controls_with_gradient_potential(blue_controls, grad_blue, η, ε)
        red_controls  = update_controls_with_gradient_potential(red_controls, grad_red, η, ε)
    end

    # === OUTPUT: Return first control for both agents ===
    blue_a_t, blue_a_r = blue_controls[1]
    red_a_t,  red_a_r  = red_controls[1]

    return blue_a_t, blue_a_r, red_a_t, red_a_r
end




function plot_results(
    blue_traj, red_traj, blue_est, red_est_by_blue,
    blue_est_by_red, blue_cov_list, red_cov_list, red_cov_by_red
)
    figure(figsize=(9, 9))
    θs = range(0, 2π; length=300)
    plot((R_inner) * cos.(θs), (R_inner) * sin.(θs), "k--", linewidth=1.0)
    plot((R_outer) * cos.(θs), (R_outer) * sin.(θs), "k--", linewidth=1.0)
    plot((R_inner + track_width / 2) * cos.(θs), (R_inner + track_width / 2) * sin.(θs), "k:", linewidth=0.8)

    # === Light Zones ===
    drew_light_zone = false
    for zone in light_zones
        θ_range = range(zone.center - zone.half_angle, zone.center + zone.half_angle; length=50)
        r0 = zone.side == "inner" ? R_inner : R_outer - light_width
        r1 = r0 + light_width
        xarc0 = r0 .* cos.(θ_range)
        yarc0 = r0 .* sin.(θ_range)
        xarc1 = r1 .* cos.(θ_range)
        yarc1 = r1 .* sin.(θ_range)

        xs = vcat(xarc0, reverse(xarc1))
        ys = vcat(yarc0, reverse(yarc1))
        fill(xs, ys, color="orange", alpha=0.3)

        if !drew_light_zone
            plot(NaN, NaN, color="orange", alpha=0.3, label="Light Zones")
            drew_light_zone = true
        end
    end

    # === Trajectories ===
    plot([p[1] for p in blue_traj], [p[2] for p in blue_traj], "b-", label="Blue True")
    plot([p[1] for p in red_traj], [p[2] for p in red_traj], "r-", label="Red True")
    plot([p[1] for p in blue_est], [p[2] for p in blue_est], "b--", label="Blue Est.")
    plot([p[1] for p in red_est_by_blue], [p[2] for p in red_est_by_blue], "r--", label="Red Est. by Blue")
    plot([p[1] for p in blue_est_by_red], [p[2] for p in blue_est_by_red], "g-.", label="Blue Est. by Red")

    # === Station markers and separation lines ===
    station_stride = 10
    for i in 1:station_stride:min(length(blue_traj), length(red_traj))
        bx, by = blue_traj[i]
        rx, ry = red_traj[i]
        plot(bx, by, "bo", markersize=4, markerfacecolor="white")
        plot(rx, ry, "rs", markersize=4, markerfacecolor="white")
        plot([bx, rx], [by, ry], "k--", linewidth=0.5, alpha=0.5)
    end

    # === Covariance ellipses ===
    function draw_ellipse(x, y, P, color)
        vals, vecs = eigen(Symmetric(P))
        t = range(0, 2π, length=50)
        circle = [cos.(t)'; sin.(t)']
        shape = vecs * diagm(sqrt.(vals)) * circle
        plot(x .+ shape[1, :], y .+ shape[2, :], color=color, alpha=0.4)
    end

    for i in 1:10:min(length(blue_cov_list), length(red_cov_list), length(red_cov_by_red))
        bx, by = blue_est[i]
        draw_ellipse(bx, by, blue_cov_list[i], "blue")

        rx, ry = red_est_by_blue[i]
        draw_ellipse(rx, ry, red_cov_list[i], "red")

        bx2, by2 = blue_est_by_red[i]
        draw_ellipse(bx2, by2, red_cov_by_red[i], "green")
    end

    legend()
    axis("equal")
    title("Potential Differential Game Racing Simulation (325759173,346395064)")
    grid(true)
    tight_layout()
    savefig("racing_differential_game_pdg.pdf", dpi=300)
end

# Function: analyze_cognitive_behavior
# ------------------------------------
# Inputs:
#   log         - CognitiveLog struct with recorded data
#   agent_name  - String indicating which agent to analyze ("blue" or "red")
# Output:
#   Saves PDF plot with subplots showing radial acceleration, estimation error,
#   and longitudinal distance to opponent
# Purpose:
#   Visualizes how each agent’s control decisions evolve based on its estimation
#   of the opponent and their relative track position.
function analyze_cognitive_behavior(log::CognitiveLog, agent_name::String)
    figure(figsize=(10, 5))
    subplot(3, 1, 1)
    plot(log.t, log.blue_ar, label="Blue a_r")
    plot(log.t, log.red_ar, label="Red a_r")
    ylabel("a_r [m/s²]")
    legend()
    grid(true)

    subplot(3, 1, 2)
    if agent_name == "blue"
        plot(log.t, log.delta_r_est, "b-", label="r_red_est - r_blue")
    else
        plot(log.t, log.delta_r_est_red, "r-", label="r_blue_est - r_red")
    end
    ylabel("Δr_est [m]")
    legend()
    grid(true)

    subplot(3, 1, 3)
    plot(log.t, log.delta_s, "k-", label="s_red - s_blue")
    xlabel("Time [s]")
    ylabel("Δs")
    legend()
    grid(true)
    tight_layout()
    suptitle("Cognitive Behavior Analysis (325759173,346395064) – $(uppercase(agent_name))", y=1.02)
    savefig("Cognitive_Behavior_Analysis_pdg.pdf", dpi=300)
end

# Function: analyze_cognitive_behavior_dual
# -----------------------------------------
# Inputs:
#   log - CognitiveLog struct
# Output:
#   Two scatter plots showing correlation between estimation error and radial
#   acceleration for both agents, color-coded by longitudinal offset
# Purpose:
#   Quantifies and visualizes the cognitive interaction between agents by
#   correlating control decisions with belief-driven distance estimations.
function analyze_cognitive_behavior_dual(log::CognitiveLog)
    # --- Blue's cognitive response ---
    Δr_b = collect(log.delta_r_est)
    ar_b = collect(log.blue_ar)
    Δs_b = collect(log.delta_s)
    ρ_b = cor(Δr_b, ar_b)

    fig1, ax1 = subplots()
    scatter1 = ax1.scatter(Δr_b, ar_b, c=Δs_b, cmap="coolwarm", edgecolors="k", alpha=0.7)
    ax1.set_xlabel("Blue’s Estimate: r_red - r_blue [m]")
    ax1.set_ylabel("Blue's Radial Acceleration a_r [m/s²]")
    ax1.set_title("Blue's Cognitive Response (325759173,346395064) (ρ = $(round(ρ_b, digits=3)))")
    fig1.colorbar(scatter1, ax=ax1, label="Δs = s_red - s_blue")
    ax1.grid(true)
    tight_layout()
    savefig("blue_cognitive_behavior_pdg.pdf", dpi=300)

    # --- Red's cognitive response ---
    Δr_r = collect(log.delta_r_est_red)
    ar_r = collect(log.red_ar)
    Δs_r = collect(log.delta_s)
    ρ_r = cor(Δr_r, ar_r)

    fig2, ax2 = subplots()
    scatter2 = ax2.scatter(Δr_r, ar_r, c=Δs_r, cmap="coolwarm", edgecolors="k", alpha=0.7)
    ax2.set_xlabel("Red’s Estimate: r_blue - r_red [m]")
    ax2.set_ylabel("Red's Radial Acceleration a_r [m/s²]")
    ax2.set_title("Red's Cognitive Response (325759173,346395064) (ρ = $(round(ρ_r, digits=3)))")
    fig2.colorbar(scatter2, ax=ax2, label="Δs = s_red - s_blue")
    ax2.grid(true)
    tight_layout()
    savefig("red_cognitive_behavior_pdg.pdf", dpi=300)

    println("Blue ρ = ", round(ρ_b, digits=3))
    println("Red  ρ = ", round(ρ_r, digits=3))
end

# Function: compute_collision_penalty
# -----------------------------------
# Inputs:
#   blue_belief - Belief of agent 1
#   red_belief  - Belief of agent 2
# Output:
#   Scalar penalty value representing expected collision risk
# Purpose:
#   Computes a soft collision penalty based on Mahalanobis distance between
#   two Gaussian beliefs. Used to discourage predicted overlap in planning.
function compute_collision_penalty(blue_belief::AgentBelief, red_belief::AgentBelief)
    μ_self = blue_belief.mean[1:2]
    Σ_self = blue_belief.cov[1:2, 1:2]
    μ_other = red_belief.mean[1:2]
    Σ_other = red_belief.cov[1:2, 1:2]
    Δμ = μ_other - μ_self
    Σ_sum = Σ_self + Σ_other + 1e-6I
    D_c_squared = dot(Δμ, Σ_sum \ Δμ)
    penalty = exp(-0.5 * D_c_squared / λ_collision)
    return penalty
end

# Function: plot_collision_penalties
# ----------------------------------
# Inputs:
#   log - CognitiveLog with time vector and collision penalty history
# Output:
#   Saves a plot showing penalty values computed by each agent
# Purpose:
#   Visualizes how each agent perceives collision risk throughout the race.
function plot_collision_penalties(log::CognitiveLog)
if length(log.t) == length(log.collision_penalty_blue) && length(log.t) == length(log.collision_penalty_red)
    figure()
    plot(log.t, log.collision_penalty_blue, label="Blue's View", color="black")
    plot(log.t, log.collision_penalty_red, label="Red's View", color="red", linestyle="--")
    title("Collision Penalty Over Time (325759173,346395064)")
    xlabel("Time [s]")
    ylabel("Penalty")
    legend()
    grid(true)
    tight_layout()
    savefig("collision_penalties_pdg.pdf", dpi=300)
else
    println("Skipping plot: collision penalty logs mismatched or missing.")
end
end

# Function: plot_radial_estimation_error
# --------------------------------------
# Inputs:
#   log - CognitiveLog with radial error vectors for both agents
# Output:
#   Saves plot showing Δr_est evolution for both blue and red perspectives
# Purpose:
#   Highlights asymmetry or agreement in agents' perception of relative radial offset.
function plot_radial_estimation_error(log::CognitiveLog)
    figure()
    plot(log.t, log.delta_r_est, "b-", label="r_red_est - r_blue")
    plot(log.t, log.delta_r_est_red, "r--", label="r_blue_est - r_red")
    xlabel("Time [s]")
    ylabel("Δr_est [m]")
    title("Radial Estimation Error (325759173,346395064)")
    legend()
    grid(true)
    tight_layout()
    savefig("radial_estimation_error_pdg.pdf", dpi=300)
end

# Function: plot_relative_distance
# --------------------------------
# Inputs:
#   log - CognitiveLog containing Δs vector
# Output:
#   Saves a time history plot of s_red - s_blue
# Purpose:
#   Shows which agent is ahead and by how much at each time step.
function plot_relative_distance(log::CognitiveLog)
    figure()
    plot(log.t, log.delta_s, "k-", label="s_red - s_blue")
    xlabel("Time [s]")
    ylabel("Δs [m]")
    title("Relative Longitudinal Distance (325759173,346395064)")
    legend()
    grid(true)
    tight_layout()
    savefig("relative_distance_pdg.pdf", dpi=300)
end

# Function: plot_acceleration_vs_estimation
# -----------------------------------------
# Inputs:
#   log - CognitiveLog with radial accelerations and estimation errors
# Output:
#   Saves a scatter plot comparing a_r with Δr_est for both agents
# Purpose:
#   Examines the control sensitivity to perceived radial offset.
function plot_acceleration_vs_estimation(log::CognitiveLog)
    figure()
    scatter(log.delta_r_est, log.blue_ar, color="blue", label="Blue")
    scatter(log.delta_r_est_red, log.red_ar, color="red", label="Red")
    xlabel("Δr_est")
    ylabel("a_r [m/s²]")
    title("Radial Acceleration vs Estimation Error (325759173,346395064)")
    legend()
    grid(true)
    tight_layout()
    savefig("ar_vs_estimation_error_pdg.pdf", dpi=300)
end

# === MAIN SIMULATION FUNCTION ===
# Function: main
# --------------------------
# Inputs:
#   (none explicitly passed; assumes access to global parameters and initial states/beliefs)
#
# Outputs:
#   Returns the following in order:
#   - log                 :: CognitiveLog         → contains agent interactions, cognition, and penalty traces
#   - blue_true_traj      :: Vector{Tuple}        → blue agent's actual trajectory
#   - red_true_traj       :: Vector{Tuple}        → red agent's actual trajectory
#   - blue_est_traj       :: Vector{Tuple}        → blue agent’s belief about its own trajectory
#   - red_est_by_blue_traj:: Vector{Tuple}        → blue’s belief about red’s trajectory
#   - blue_est_by_red_traj:: Vector{Tuple}        → red’s belief about blue’s trajectory
#   - blue_cov_list       :: Vector{Matrix}       → covariance of blue’s self-belief over time
#   - red_cov_list        :: Vector{Matrix}       → blue's belief about red's covariance
#   - red_cov_by_red      :: Vector{Matrix}       → red’s own belief about red’s covariance
#
# Purpose:
#   This is the core simulation loop for a belief-space autonomous racing scenario.
#   It simulates a differential game where two agents (blue and red) race around a track,
#   continuously updating their beliefs about themselves and each other using EKF.
#
#   At each timestep:
#   - Each agent computes a Nash equilibrium action using belief-aware best-response optimization.
#   - Belief propagation and updates are performed (including noisy observations).
#   - Radial distance, estimation error, and collision penalties are logged for post-analysis.
#   - All relevant state and belief trajectories are stored for visualization and evaluation.
#
#   The simulation ends when both agents have completed a full loop of the track.
function main()
    log = CognitiveLog([], [], [], [], [], [], [], [])
    track_length = 2π * (R_inner + track_width / 2)

    # Trajectory logs
    blue_true_traj = Vector{Tuple{Float64, Float64}}()
    red_true_traj  = Vector{Tuple{Float64, Float64}}()
    blue_est_traj  = Vector{Tuple{Float64, Float64}}()
    red_est_by_blue_traj = Vector{Tuple{Float64, Float64}}()
    blue_est_by_red_traj = Vector{Tuple{Float64, Float64}}()
    blue_cov_list = Vector{Matrix{Float64}}()
    red_cov_list  = Vector{Matrix{Float64}}()
    red_cov_by_red = Vector{Matrix{Float64}}()

    # Blue agent's car model
    model_blue = CarModel(
        0.25,      # Moderate longitudinal resistance
        0.05      # Moderate lateral slip
    )

    # Red agent's car model
    model_red = CarModel(
        0.2,     # Slightly more drag
        0.03      # More stable laterally
    )

    t = 0.0
    while t < max_time && (blue_state.s < track_length || red_state.s < track_length)
        # === Record current states and beliefs ===
        push!(blue_true_traj, (blue_state.x, blue_state.y))
        push!(red_true_traj,  (red_state.x,  red_state.y))
        push!(blue_est_traj,  (blue_belief_self.mean[1], blue_belief_self.mean[2]))
        push!(red_est_by_blue_traj, (blue_belief_red.mean[1], blue_belief_red.mean[2]))
        push!(blue_est_by_red_traj, (red_belief_blue.mean[1], red_belief_blue.mean[2]))

        push!(blue_cov_list, blue_belief_self.cov[1:2, 1:2])
        push!(red_cov_list,  blue_belief_red.cov[1:2, 1:2])
        push!(red_cov_by_red, red_belief_blue.cov[1:2, 1:2])

        # === Differential Game Logic: Compute Nash Controls ===
        blue_a_t, blue_a_r, red_a_t, red_a_r = compute_nash_controls!(
            blue_belief_self, red_belief_self,
            blue_belief_red, red_belief_blue,
            v_des_blue, v_des_red,
            Kp_v_blue, Kp_v_red,
            model_blue=model_blue, model_red=model_red
        )

        # === Cognitive Logging ===
        push!(log.t, t)
        push!(log.blue_ar, blue_a_r)
        push!(log.red_ar, red_a_r)

        bx, by = blue_belief_self.mean[1:2]
        rx_est, ry_est = blue_belief_red.mean[1:2]
        r_blue = sqrt(bx^2 + by^2)
        r_red_est = sqrt(rx_est^2 + ry_est^2)
        push!(log.delta_r_est, r_red_est - r_blue)

        rrx, rry = red_belief_self.mean[1:2]
        bx_est, by_est = red_belief_blue.mean[1:2]
        r_red = sqrt(rrx^2 + rry^2)
        r_blue_est = sqrt(bx_est^2 + by_est^2)
        push!(log.delta_r_est_red, r_blue_est - r_red)

        push!(log.delta_s, red_state.s - blue_state.s)

        # === Propagate Dynamics ===
        propagate_state!(blue_state, blue_a_t, blue_a_r, dt)
        propagate_state!(red_state,  red_a_t,  red_a_r,  dt)

        # === Predict Beliefs ===
        belief_predict!(blue_belief_self, blue_a_t, blue_a_r, dt, model_blue)
        belief_predict!(red_belief_self,  red_a_t,  red_a_r,  dt, model_red)
        belief_predict!(blue_belief_red,  red_a_t,  red_a_r,  dt, model_red)
        belief_predict!(red_belief_blue,  blue_a_t, blue_a_r, dt, model_blue)

        # === Generate Observations ===
        R_blue = in_light_zone(blue_state.x, blue_state.y, light_zones) ?
            Diagonal([0.2^2, 0.2^2]) : Diagonal([5.0^2, 5.0^2])
        R_red = in_light_zone(red_state.x, red_state.y, light_zones) ?
            Diagonal([0.2^2, 0.2^2]) : Diagonal([5.0^2, 5.0^2])

        R_blue_on_red = in_light_zone(red_state.x, red_state.y, light_zones) ?
            Diagonal([0.2^2, 0.2^2]) : Diagonal([5.0^2, 5.0^2])
        R_red_on_blue = in_light_zone(blue_state.x, blue_state.y, light_zones) ?
            Diagonal([0.2^2, 0.2^2]) : Diagonal([5.0^2, 5.0^2])

        blue_meas = [blue_state.x, blue_state.y] .+ randn(2) .* sqrt.(diag(R_blue))
        red_meas  = [red_state.x,  red_state.y]  .+ randn(2) .* sqrt.(diag(R_red))
        red_meas_by_blue  = [red_state.x, red_state.y] .+ randn(2) .* sqrt.(diag(R_blue_on_red))
        blue_meas_by_red  = [blue_state.x, blue_state.y] .+ randn(2) .* sqrt.(diag(R_red_on_blue))

        # === EKF Belief Updates ===
        belief_update!(blue_belief_self, blue_meas, R_blue)
        belief_update!(red_belief_self,  red_meas,  R_red)
        belief_update!(blue_belief_red,  red_meas_by_blue, R_blue_on_red)
        belief_update!(red_belief_blue,  blue_meas_by_red, R_red_on_blue)

        push!(log.collision_penalty_blue, compute_collision_penalty(blue_belief_self, blue_belief_red))
        push!(log.collision_penalty_red, compute_collision_penalty(red_belief_self, red_belief_blue))

        t += dt
    end

    # Final state capture
    push!(blue_true_traj, (blue_state.x, blue_state.y))
    push!(red_true_traj,  (red_state.x, red_state.y))
    push!(blue_est_traj,  (blue_belief_self.mean[1], blue_belief_self.mean[2]))
    push!(red_est_by_blue_traj, (blue_belief_red.mean[1], blue_belief_red.mean[2]))
    push!(blue_est_by_red_traj, (red_belief_blue.mean[1], red_belief_blue.mean[2]))
    push!(blue_cov_list, blue_belief_self.cov[1:2, 1:2])
    push!(red_cov_list,  blue_belief_red.cov[1:2, 1:2])
    push!(red_cov_by_red, red_belief_blue.cov[1:2, 1:2])

    return log, blue_true_traj, red_true_traj, blue_est_traj, red_est_by_blue_traj,
        blue_est_by_red_traj, blue_cov_list, red_cov_list, red_cov_by_red
end


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# === INITIAL POSITIONS AND VELOCITIES ===
blue_start_angle = 0.2
red_start_angle  = 0
blue_start_r = R_inner + 1.0
red_start_r  = R_inner + 5.0

blue_start_x = blue_start_r * cos(blue_start_angle)
blue_start_y = blue_start_r * sin(blue_start_angle)
red_start_x  = red_start_r * cos(red_start_angle)
red_start_y  = red_start_r * sin(red_start_angle)

blue_speed0 = 10.0
red_speed0  = 10.0

blue_tangent = [-sin(blue_start_angle), cos(blue_start_angle)]
red_tangent  = [-sin(red_start_angle), cos(red_start_angle)]

blue_vx0, blue_vy0 = blue_speed0 * blue_tangent[1], blue_speed0 * blue_tangent[2]
red_vx0,  red_vy0  = red_speed0  * red_tangent[1],  red_speed0  * red_tangent[2]

# === CREATE STATE OBJECTS ===
blue_state = AgentState(blue_start_x, blue_start_y, blue_vx0, blue_vy0, 0.0)
red_state  = AgentState(red_start_x,  red_start_y,  red_vx0,  red_vy0,  5.0)


# === INIT BELIEFS ===
std_devs_self = [0.5, 0.5, 0.25, 0.25]
std_devs_opponent = [1.0, 1.0, 0.5, 0.5]

blue_belief_self = create_belief(blue_state.x, blue_state.y, blue_state.vx, blue_state.vy, std_devs_self)
red_belief_self  = create_belief(red_state.x,  red_state.y,  red_state.vx,  red_state.vy,  std_devs_self)
blue_belief_red  = create_belief(red_state.x,  red_state.y,  red_state.vx,  red_state.vy,  std_devs_opponent)
red_belief_blue  = create_belief(blue_state.x, blue_state.y, blue_state.vx, blue_state.vy, std_devs_opponent)

log, blue_traj, red_traj, blue_est, red_est_by_blue, blue_est_by_red,
blue_cov_list, red_cov_list, red_cov_by_red = main()

plot_results(blue_traj, red_traj, blue_est, red_est_by_blue, blue_est_by_red,
            blue_cov_list, red_cov_list, red_cov_by_red)

analyze_cognitive_behavior(log, "blue")
analyze_cognitive_behavior(log, "red")

# === Analyze cognitive responses of both agents - correlation between control and Δr ===
analyze_cognitive_behavior_dual(log)

plot_collision_penalties(log)
plot_radial_estimation_error(log)
plot_relative_distance(log)
plot_acceleration_vs_estimation(log)



