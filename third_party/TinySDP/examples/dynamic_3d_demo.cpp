#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <solver/tiny_api.hpp>
#include <solver/psd_support.hpp>

#ifndef TINYSDP_SCENARIO_SLUG
#define TINYSDP_SCENARIO_SLUG "sweeping_barrier"
#endif

namespace {

constexpr int NX0 = 6;   // x, y, z, vx, vy, vz
constexpr int NU0 = 3;   // ax, ay, az
constexpr int N   = 10;

using Mat = Eigen::Matrix<tinytype, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Matrix<tinytype, Eigen::Dynamic, 1>;

struct MovingSphere {
    tinytype cx0;
    tinytype cy0;
    tinytype cz0;
    tinytype vx;
    tinytype vy;
    tinytype vz;
    tinytype radius;
    tinytype wobble_x;
    tinytype wobble_x_freq;
    tinytype wobble_x_phase;
    tinytype wobble_y;
    tinytype wobble_y_freq;
    tinytype wobble_y_phase;
    tinytype wobble_z;
    tinytype wobble_z_freq;
    tinytype wobble_z_phase;

    std::array<tinytype, 4> sphere_at_time(tinytype t) const {
        double td = static_cast<double>(t);
        double xarg = static_cast<double>(wobble_x_freq) * td + static_cast<double>(wobble_x_phase);
        double yarg = static_cast<double>(wobble_y_freq) * td + static_cast<double>(wobble_y_phase);
        double zarg = static_cast<double>(wobble_z_freq) * td + static_cast<double>(wobble_z_phase);
        tinytype cx = cx0 + vx * t + wobble_x * tinytype(std::sin(xarg));
        tinytype cy = cy0 + vy * t + wobble_y * tinytype(std::cos(yarg));
        tinytype cz = cz0 + vz * t + wobble_z * tinytype(std::sin(zarg));
        return {cx, cy, cz, radius};
    }
};

bool sphere_has_motion(const MovingSphere& s) {
    auto nz = [](tinytype v) { return std::abs(static_cast<double>(v)) > 1e-9; };
    return nz(s.vx) || nz(s.vy) || nz(s.vz)
        || nz(s.wobble_x) || nz(s.wobble_y) || nz(s.wobble_z)
        || nz(s.wobble_x_freq) || nz(s.wobble_y_freq) || nz(s.wobble_z_freq);
}

struct DynamicSpheres {
    std::vector<MovingSphere> agents;
    tinytype dt = tinytype(1.0);

    std::vector<std::array<tinytype, 4>> spheres_at_step(int step) const {
        tinytype t = dt * tinytype(step);
        std::vector<std::array<tinytype, 4>> spheres;
        spheres.reserve(agents.size());
        for (const auto& agent : agents) {
            spheres.push_back(agent.sphere_at_time(t));
        }
        return spheres;
    }

    std::vector<std::vector<std::array<tinytype, 4>>> horizon_spheres_per_stage(
        int step,
        int horizon,
        tinytype inflation_rate) const {
        std::vector<std::vector<std::array<tinytype, 4>>> per_stage;
        per_stage.reserve(horizon);
        for (int h = 0; h < horizon; ++h) {
            auto spheres = spheres_at_step(step + h);
            tinytype inflate = inflation_rate * tinytype(std::sqrt(static_cast<double>(h)));
            for (auto& s : spheres) {
                s[3] += inflate;
            }
            per_stage.push_back(std::move(spheres));
        }
        return per_stage;
    }
};

struct ScenarioSpec {
    std::string name;
    std::string slug;
    DynamicSpheres obstacles;
    std::vector<Eigen::Vector3d> guide_points;
    tinytype prediction_inflation = tinytype(0.02);
    double activation_on = 0.75;
    double activation_off = 0.95;
    int total_steps = 28;
};

struct PsdCertificate {
    double trace_gap;
    double eta_min;
    double true_dist2_min;
    bool certified;
};

struct PlanCache {
    std::vector<Vec> states;
    std::vector<Vec> inputs;
    int start_step = 0;
    int last_iters = 0;
};

struct ScenarioResult {
    std::string name;
    std::string slug;
    bool success = false;
    int goal_step = -1;
    double min_point_sd = std::numeric_limits<double>::infinity();
    double min_seg_sd = std::numeric_limits<double>::infinity();
    double min_eta = std::numeric_limits<double>::infinity();
    double max_trace_gap = 0.0;
    bool all_certified = true;
    double final_goal_dist = std::numeric_limits<double>::infinity();
    double final_vel_norm = std::numeric_limits<double>::infinity();
    int planner_solves = 0;
    int tracker_solves = 0;
    double planner_total_us = 0.0;
    double tracker_total_us = 0.0;
};

struct RunConfig {
    int replan_stride = N - 1;
    std::string file_prefix = "tinysdp_3d_";
    bool log_plan_states = false;
    bool mocap_mode = false;
    std::vector<std::string> scenario_filter;
};

Vec build_lifted(const Vec& base_state) {
    const int nxL = NX0 + NX0 * NX0;
    Vec lifted(nxL);
    lifted.setZero();
    lifted.topRows(NX0) = base_state;
    Mat outer = base_state * base_state.transpose();
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            lifted(NX0 + j * NX0 + i) = outer(i, j);
        }
    }
    return lifted;
}

double signed_distance_point_spheres(const Vec& x,
                                     const std::vector<std::array<tinytype,4>>& spheres) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& s : spheres) {
        double dx = static_cast<double>(x(0) - s[0]);
        double dy = static_cast<double>(x(1) - s[1]);
        double dz = static_cast<double>(x(2) - s[2]);
        double r  = static_cast<double>(s[3]);
        best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - r);
    }
    return best;
}

double signed_distance_segment_spheres(const Vec& p0,
                                       const Vec& p1,
                                       const std::vector<std::array<tinytype,4>>& spheres) {
    double best = std::numeric_limits<double>::infinity();
    double x0 = static_cast<double>(p0(0));
    double y0 = static_cast<double>(p0(1));
    double z0 = static_cast<double>(p0(2));
    double x1 = static_cast<double>(p1(0));
    double y1 = static_cast<double>(p1(1));
    double z1 = static_cast<double>(p1(2));
    double ddx = x1 - x0;
    double ddy = y1 - y0;
    double ddz = z1 - z0;
    double len2 = ddx * ddx + ddy * ddy + ddz * ddz;
    for (const auto& s : spheres) {
        double cx = static_cast<double>(s[0]);
        double cy = static_cast<double>(s[1]);
        double cz = static_cast<double>(s[2]);
        double r  = static_cast<double>(s[3]);
        double t = 0.0;
        if (len2 > 0.0) {
            t = ((cx - x0) * ddx + (cy - y0) * ddy + (cz - z0) * ddz) / len2;
            t = std::max(0.0, std::min(1.0, t));
        }
        double px = x0 + t * ddx;
        double py = y0 + t * ddy;
        double pz = z0 + t * ddz;
        double sd = std::sqrt((px - cx) * (px - cx) +
                              (py - cy) * (py - cy) +
                              (pz - cz) * (pz - cz)) - r;
        best = std::min(best, sd);
    }
    return best;
}

PsdCertificate compute_psd_certificate(const Vec& x_lifted,
                                       const std::vector<std::array<tinytype,4>>& spheres) {
    PsdCertificate cert;
    cert.trace_gap = 0.0;
    cert.eta_min = std::numeric_limits<double>::infinity();
    cert.true_dist2_min = std::numeric_limits<double>::infinity();
    cert.certified = false;

    Eigen::Vector3d z;
    z << static_cast<double>(x_lifted(0)),
         static_cast<double>(x_lifted(1)),
         static_cast<double>(x_lifted(2));

    Eigen::Matrix<double, NX0, NX0> XX_full;
    for (int j = 0; j < NX0; ++j) {
        for (int i = 0; i < NX0; ++i) {
            XX_full(i, j) = static_cast<double>(x_lifted(NX0 + j * NX0 + i));
        }
    }

    Eigen::Matrix3d XX = XX_full.topLeftCorner<3, 3>();
    cert.trace_gap = XX.trace() - z.squaredNorm();

    for (const auto& s : spheres) {
        Eigen::Vector3d c;
        c << static_cast<double>(s[0]),
             static_cast<double>(s[1]),
             static_cast<double>(s[2]);
        double r = static_cast<double>(s[3]);
        double lifted_dist2 = XX.trace() - 2.0 * c.dot(z) + c.squaredNorm();
        double eta = lifted_dist2 - r * r;
        double true_dist2 = (z - c).squaredNorm() - r * r;
        cert.eta_min = std::min(cert.eta_min, eta);
        cert.true_dist2_min = std::min(cert.true_dist2_min, true_dist2);
    }

    cert.certified = (cert.eta_min >= 0.0) && (std::abs(cert.trace_gap) <= cert.eta_min);
// ---- D1 (env TINYSDP_FIX_CERT=1): an honest certificate ----------------
    // The shipped test is  certified = (eta>=0) && (|D| <= eta).  But
    //     eta - D = ||z||^2 - 2c.z + ||c||^2 - r^2 = ||z-c||^2 - r^2,
    // so tr(XX) cancels and the test is just "point outside sphere": it passes
    // with D = 4485.  The two things actually being claimed are separate:
    //   (a) the relaxation is tight   ->  |D| <= eps_rank
    //   (b) the recovered point clears ->  eta - D >= 0
    // and (b) is the exact true margin, which is what should be reported.
    if (std::getenv("TINYSDP_FIX_CERT")) {
        const double eps_rank = 1e-6 * (1.0 + z.squaredNorm());
        cert.certified = (std::abs(cert.trace_gap) <= eps_rank) &&
                         (cert.true_dist2_min >= 0.0);
    }
    return cert;
}

int clamp_index(int idx, int lo, int hi) {
    if (idx < lo) return lo;
    if (idx > hi) return hi;
    return idx;
}

Eigen::Vector3d sample_polyline(const std::vector<Eigen::Vector3d>& route, double alpha) {
    if (route.empty()) return Eigen::Vector3d::Zero();
    if (route.size() == 1) return route.front();
    double s = std::max(0.0, std::min(1.0, alpha));
    double scaled = s * static_cast<double>(route.size() - 1);
    int idx = static_cast<int>(std::floor(scaled));
    if (idx >= static_cast<int>(route.size()) - 1) {
        return route.back();
    }
    double local = scaled - static_cast<double>(idx);
    return (1.0 - local) * route[idx] + local * route[idx + 1];
}

Vec clamp_input(const Vec& u, tinytype limit) {
    Vec out = u;
    for (int i = 0; i < out.rows(); ++i) {
        out(i) = std::max(-limit, std::min(limit, out(i)));
    }
    return out;
}

void rollout_plan(const Mat& Ad,
                  const Mat& Bd,
                  const Vec& x_start,
                  TinySolver* solver,
                  PlanCache* cache) {
    cache->states.assign(N, Vec::Zero(NX0));
    cache->inputs.assign(N - 1, Vec::Zero(NU0));
    Vec x = x_start;
    cache->states[0] = x;
    for (int k = 0; k < N - 1; ++k) {
        Vec u = solver->solution->u.col(k).topRows(NU0);
        cache->inputs[k] = u;
        x = Ad * x + Bd * u;
        cache->states[k + 1] = x;
    }
    cache->last_iters = solver->solution->iter;
}

void set_base_tracking_refs(TinySolver* solver,
                            const PlanCache& cache,
                            int current_step) {
    Mat Xref = Mat::Zero(NX0, N);
    Mat Uref = Mat::Zero(NU0, N - 1);
    if (!cache.states.empty()) {
        const int max_idx = static_cast<int>(cache.states.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx);
            Xref.col(i) = cache.states[idx];
        }
    }
    if (!cache.inputs.empty()) {
        const int max_idx_u = static_cast<int>(cache.inputs.size()) - 1;
        int offset = current_step - cache.start_step;
        for (int i = 0; i < N - 1; ++i) {
            int idx = clamp_index(offset + i, 0, max_idx_u);
            Uref.col(i) = cache.inputs[idx];
        }
    }
    tiny_set_x_ref(solver, Xref);
    tiny_set_u_ref(solver, Uref);
}

// ============================ D4: DELETE THE CONE ============================
// Rank one is not an extra assumption here, it is what the relaxation is trying
// to reach.  Impose it and the whole lift collapses:
//     XX = z z^T,  XU = z w^T,  UU = w w^T
// makes the lifted dynamics (A(x)A, the Kronecker B blocks) automatically
// consistent, and the lifted obstacle row
//     tr(XX) - 2 c.z + |c|^2 >= r^2
// becomes exactly  ||z - c||^2 >= r^2.  The 42-vector state, the 48-vector
// input, the 10x10 cone, its projection, its slack S and its dual H all vanish.
// What is left is the ORIGINAL 6-state problem with an exact ball-complement
// constraint, which we handle by sequential linearisation about the current
// iterate:
//     n_kj . (z_k - c_j) >= r_j,     n_kj = (z_k - c_j)/||z_k - c_j||.
// By Cauchy-Schwarz that half-plane lies INSIDE the ball complement, so it is a
// conservative inner approximation: a point that satisfies it clears the ball
// exactly.  There is no relaxation gap, hence nothing to certify away -- the
// plan's clearance IS the true clearance.
// ================= D4-N: exact second-order inner solve =================
// D4 removed the cone but kept their first-order ADMM, which is where the cost
// actually sits (measured: with cuts on it never converges, 500 iterations at a
// primal residual of 0.8-3.0 m).  Replace it outright.
//
// Two structural moves, both exact:
//   1. CONDENSE.  Eliminate the dynamics by substitution, z = A^k z0 + S w.
//      The 105 primal variables and 72 equality rows collapse to 33 free input
//      variables with NO equality constraints, and the Hessian S'QS + R is
//      dense but positive definite.
//   2. ACTIVE SET + KKT.  Solve the resulting inequality-constrained QP exactly
//      by a primal-dual active set: solve the equality-constrained KKT on the
//      working set, drop constraints whose multiplier turns negative, add the
//      most violated one, repeat.  Same logic as P2's ChildActiveSet.
// The subproblem is then solved to machine precision, so the plan satisfies the
// requested margin exactly instead of approximately.
struct D4N {
    int nw = 0, nz = 0;
    Mat S, Apow, Hc, state_lo, state_hi;
    Vec qdiag;
    double ulim = 3.0;
    bool ready = false;
};

static void d4n_build(D4N& s, const Mat& Ad, const Mat& Bd,
                      const Mat& Qd, const Mat& Rd, double ulim)
{
    s.nz = NX0 * N; s.nw = NU0 * (N - 1); s.ulim = ulim;
    s.S = Mat::Zero(s.nz, s.nw);
    s.Apow = Mat::Zero(s.nz, NX0);
    std::vector<Mat> Ak(N);
    Mat P = Mat::Identity(NX0, NX0);
    for (int k = 0; k < N; ++k) { Ak[k] = P; s.Apow.block(k*NX0,0,NX0,NX0) = P; P = Ad * P; }
    for (int k = 1; k < N; ++k)
        for (int j = 0; j < k; ++j)
            s.S.block(k*NX0, j*NU0, NX0, NU0) = Ak[k-1-j] * Bd;
    s.qdiag = Vec::Zero(s.nz);
    for (int k = 0; k < N; ++k) s.qdiag.segment(k*NX0, NX0) = Qd.diagonal();
    Mat QS = s.S;
    for (int i = 0; i < s.nz; ++i) QS.row(i) *= s.qdiag(i);
    s.Hc = s.S.transpose() * QS;
    for (int j = 0; j < N - 1; ++j) s.Hc.block(j*NU0, j*NU0, NU0, NU0) += Rd;
    s.ready = true;
}

// Solve  min 0.5 w'Hc w + gc'w  s.t. Ain w <= bin, exactly.
static int d4n_qp(const D4N& s, const Vec& gc, const Mat& Ain, const Vec& bin,
                  Vec& w, int max_as = 60)
{
    const int nw = s.nw, m = int(Ain.rows());
    std::vector<char> inW(m, 0);
    int solves = 0;
    const double tol = 1e-9;
    for (int pass = 0; pass < max_as; ++pass) {
        std::vector<int> W;
        for (int i = 0; i < m; ++i) if (inW[i]) W.push_back(i);
        const int nA = int(W.size());
        Mat K = Mat::Zero(nw + nA, nw + nA);
        Vec rhs = Vec::Zero(nw + nA);
        K.topLeftCorner(nw, nw) = s.Hc;
        rhs.head(nw) = -gc;
        for (int a = 0; a < nA; ++a) {
            K.block(nw + a, 0, 1, nw) = Ain.row(W[a]);
            K.block(0, nw + a, nw, 1) = Ain.row(W[a]).transpose();
            rhs(nw + a) = bin(W[a]);
        }
        Eigen::PartialPivLU<Mat> lu(K);
        Vec sol = lu.solve(rhs);
        solves++;
        if (!sol.allFinite()) return -solves;
        if (std::getenv("TINYSDP_REVISION") &&
            (K * sol - rhs).lpNorm<Eigen::Infinity>() >
                1e-8 * (1.0 + rhs.lpNorm<Eigen::Infinity>())) return -solves;
        w = sol.head(nw);
        // drop: a working constraint pushing the wrong way
        int drop = -1; double worst = -tol;
        for (int a = 0; a < nA; ++a) {
            double lam = double(sol(nw + a));
            if (lam < worst) { worst = lam; drop = W[a]; }
        }
        if (drop >= 0) { inW[drop] = 0; continue; }
        // add: the most violated inactive constraint
        int add = -1; double vmax = 1e-8;
        for (int i = 0; i < m; ++i) {
            if (inW[i]) continue;
            double v = double(Ain.row(i).dot(w)) - double(bin(i));
            if (v > vmax) { vmax = v; add = i; }
        }
        if (add < 0) return solves;   // optimal
        inW[add] = 1;
    }
    return -solves;
}
// =============== END D4-N ===============

struct D4Out { int outer = 0; int iters_total = 0; bool feasible = true; };

static void d4_set_cuts(TinySolver* s, int rows_per_stage,
                        const std::vector<std::vector<std::array<tinytype,4>>>& sph,
                        const std::vector<Vec>& Z)
{
    if (rows_per_stage <= 0) {
        s->work->tv_Alin_x.setZero();
        s->work->tv_blin_x.setConstant(tinytype(1e6));
        return;
    }
    const int Nn = s->work->N;
    const tinytype relaxed_upper = tinytype(1e6);
    // One cut per stage by default: the nearest sphere.  Linearising several
    // spheres at the SAME point produces half-planes that push in opposing
    // directions -- with a wall of spheres the intersection is empty and the
    // inner ADMM cannot converge (measured: 500 iterations, primal residual
    // stuck at 0.8-3.0 m).  The nearest sphere is the only one that can bind.
    const bool one_cut = (std::getenv("TINYSDP_D4_ALLCUTS") == nullptr);
    for (int k = 0; k < Nn; ++k) {
        const std::vector<std::array<tinytype,4>> empty_stage;
        const auto& st_all = (k < static_cast<int>(sph.size())) ? sph[k] : empty_stage;
        std::vector<std::array<tinytype,4>> st;
        if (one_cut && !st_all.empty()) {
            int best = 0; double bestd = std::numeric_limits<double>::infinity();
            for (std::size_t q = 0; q < st_all.size(); ++q) {
                double dx = double(Z[k](0) - st_all[q][0]);
                double dy = double(Z[k](1) - st_all[q][1]);
                double dz3 = double(Z[k](2) - st_all[q][2]);
                double d = std::sqrt(dx*dx + dy*dy + dz3*dz3) - double(st_all[q][3]);
                if (d < bestd) { bestd = d; best = int(q); }
            }
            st.push_back(st_all[best]);
        } else {
            st = st_all;
        }
        for (int j = 0; j < rows_per_stage; ++j) {
            const int row = k * rows_per_stage + j;
            if (j < static_cast<int>(st.size())) {
                Eigen::Vector3d c(static_cast<double>(st[j][0]),
                                  static_cast<double>(st[j][1]),
                                  static_cast<double>(st[j][2]));
                // An exact constraint is respected exactly, so the planner will
                // ride the boundary unless a margin is asked for.  The relaxed
                // version had a large margin only by accident, from its own
                // looseness.  Here it is explicit and auditable.
                static const char* infs = std::getenv("TINYSDP_D4_INFLATE");
                static const double inflate = infs ? atof(infs) : 0.0;
                const double r = static_cast<double>(st[j][3]) + inflate;
                Eigen::Vector3d pt(static_cast<double>(Z[k](0)),
                                   static_cast<double>(Z[k](1)),
                                   static_cast<double>(Z[k](2)));
                Eigen::Vector3d d = pt - c;
                const double nd = d.norm();
                Eigen::Vector3d n = (nd > 1e-9) ? Eigen::Vector3d(d / nd)
                                                : Eigen::Vector3d(1.0, 0.0, 0.0);
                // n.(z - c) >= r   <=>   (-n).z <= -(r + n.c)
                tinyVector a = tinyVector::Zero(s->work->nx);
                a(0) = tinytype(-n.x()); a(1) = tinytype(-n.y()); a(2) = tinytype(-n.z());
                s->work->tv_Alin_x.row(row) = a.transpose();
                s->work->tv_blin_x(j, k) = tinytype(-(r + n.dot(c)));
            } else {
                s->work->tv_Alin_x.row(row).setZero();
                s->work->tv_blin_x(j, k) = relaxed_upper;
            }
        }
    }
}

// SCP driver on the exact inner solve.  Same outer logic as d4_plan (nearest
// sphere per stage, damped linearisation point) so the only thing that changes
// between the two rows of the results table is HOW the subproblem is solved.
static D4Out d4_plan_newton(const D4N& s, const Vec& x_seed, const Mat& Xref6,
                            const std::vector<std::vector<std::array<tinytype,4>>>& sph,
                            std::vector<Vec>* states, std::vector<Vec>* inputs)
{
    D4Out o;
    const int nw = s.nw, nz = s.nz;
    Vec zf = s.Apow * x_seed;
    Vec zref = Vec::Zero(nz);
    for (int k = 0; k < N; ++k) zref.segment(k*NX0, NX0) = Xref6.col(k);
    Vec gc = s.S.transpose() * (s.qdiag.array() * (zf - zref).array()).matrix();

    std::vector<Vec> Z(N, Vec::Zero(NX0));
    for (int k = 0; k < N; ++k) Z[k] = Xref6.col(k);
    Z[0] = x_seed;

    const char* infs = std::getenv("TINYSDP_D4_INFLATE");
    const double inflate = infs ? atof(infs) : 0.0;
    const char* as_ = std::getenv("TINYSDP_D4_ALPHA");
    const double alpha = as_ ? atof(as_) : 0.5;
    const char* mo = std::getenv("TINYSDP_D4_OUTER");
    const int max_outer = mo ? atoi(mo) : 8;

    Vec w = Vec::Zero(nw);
    states->assign(N, Vec::Zero(NX0));
    inputs->assign(N - 1, Vec::Zero(NU0));

    // COMMITMENT MODE (TINYSDP_D4_COMMIT=1).  P2's move, transplanted: instead
    // of re-linearising the ball complement about the current iterate, commit
    // ONCE to a homotopy class -- one separating direction per OBSTACLE, taken
    // from the guide route, held at every stage.  The direction is the guide's
    // own closest approach to that obstacle, which is the tightest halfspace
    // the guide certifies.  No re-linearisation, so no outer loop at all.
    const bool commit = (std::getenv("TINYSDP_D4_COMMIT") != nullptr);
    std::vector<int> ckstage; std::vector<Eigen::Vector3d> cn; std::vector<double> cb;
    if (commit) {
        // obstacle j is the j-th entry of every stage list (stable ordering)
        std::size_t nobs = 0;
        for (const auto& v : sph) nobs = std::max(nobs, v.size());
        for (std::size_t j = 0; j < nobs; ++j) {
            int kstar = -1; double dmin = std::numeric_limits<double>::infinity();
            Eigen::Vector3d cstar(0,0,0); double rstar = 0.0;
            for (int k = 1; k < N; ++k) {
                if (k >= int(sph.size()) || j >= sph[k].size()) continue;
                Eigen::Vector3d c(static_cast<double>(sph[k][j][0]),
                                  static_cast<double>(sph[k][j][1]),
                                  static_cast<double>(sph[k][j][2]));
                Eigen::Vector3d g(static_cast<double>(Z[k](0)),
                                  static_cast<double>(Z[k](1)),
                                  static_cast<double>(Z[k](2)));
                double d = (g - c).norm();
                if (d < dmin) { dmin = d; kstar = k; cstar = c; rstar = static_cast<double>(sph[k][j][3]); }
            }
            if (kstar < 0) continue;
            Eigen::Vector3d g(static_cast<double>(Z[kstar](0)),
                              static_cast<double>(Z[kstar](1)),
                              static_cast<double>(Z[kstar](2)));
            Eigen::Vector3d d = g - cstar; double nd = d.norm();
            Eigen::Vector3d n = (nd > 1e-9) ? Eigen::Vector3d(d / nd) : Eigen::Vector3d(1,0,0);
            for (int k = 1; k < N; ++k) {
                if (k >= int(sph.size()) || j >= sph[k].size()) continue;
                Eigen::Vector3d ck(static_cast<double>(sph[k][j][0]),
                                   static_cast<double>(sph[k][j][1]),
                                   static_cast<double>(sph[k][j][2]));
                double rk = static_cast<double>(sph[k][j][3]) + inflate;
                ckstage.push_back(k); cn.push_back(n); cb.push_back(-(rk + n.dot(ck)));
            }
        }
    }

    for (int it = 0; it < max_outer; ++it) {
        // one linearised cut per stage, at the nearest sphere
        std::vector<int> kcut; std::vector<Eigen::Vector3d> ncut; std::vector<double> bcut;
        if (commit) { kcut = ckstage; ncut = cn; bcut = cb; } else
        for (int k = 1; k < N; ++k) {
            if (k >= int(sph.size()) || sph[k].empty()) continue;
            int best = 0; double bd = std::numeric_limits<double>::infinity();
            for (std::size_t q = 0; q < sph[k].size(); ++q) {
                Eigen::Vector3d c(static_cast<double>(sph[k][q][0]), static_cast<double>(sph[k][q][1]), static_cast<double>(sph[k][q][2]));
                double d = (Eigen::Vector3d(static_cast<double>(Z[k](0)),static_cast<double>(Z[k](1)),static_cast<double>(Z[k](2))) - c).norm()
                           - static_cast<double>(sph[k][q][3]);
                if (d < bd) { bd = d; best = int(q); }
            }
            const bool allcuts = std::getenv("TINYSDP_D4_ALLCUTS") != nullptr;
            for (int selected = 0; selected < int(sph[k].size()); ++selected) {
                if (!allcuts && selected != best) continue;
            Eigen::Vector3d c(static_cast<double>(sph[k][selected][0]), static_cast<double>(sph[k][selected][1]), static_cast<double>(sph[k][selected][2]));
            double r = static_cast<double>(sph[k][selected][3]) + inflate;
            Eigen::Vector3d d(static_cast<double>(Z[k](0)) - c.x(), static_cast<double>(Z[k](1)) - c.y(), static_cast<double>(Z[k](2)) - c.z());
            double nd = d.norm();
            Eigen::Vector3d n = (nd > 1e-9) ? Eigen::Vector3d(d / nd) : Eigen::Vector3d(1,0,0);
            kcut.push_back(k); ncut.push_back(n); bcut.push_back(-(r + n.dot(c)));
            }
        }
        const int mcut = int(kcut.size());
        const bool revision_bounds = std::getenv("TINYSDP_REVISION") != nullptr;
        const int mstate = revision_bounds ? 2 * NX0 * (N-1) : 0;
        const int m = mcut + 2 * nw + mstate;
        Mat Ain = Mat::Zero(m, nw);
        Vec bin = Vec::Zero(m);
        for (int i = 0; i < mcut; ++i) {
            const int k = kcut[i];
            Vec a6 = Vec::Zero(NX0);
            a6(0) = tinytype(-ncut[i].x()); a6(1) = tinytype(-ncut[i].y()); a6(2) = tinytype(-ncut[i].z());
            Ain.row(i) = (s.S.block(k*NX0, 0, NX0, nw).transpose() * a6).transpose();
            bin(i) = tinytype(bcut[i]) - a6.dot(zf.segment(k*NX0, NX0));
        }
        for (int i = 0; i < nw; ++i) {
            Ain(mcut + 2*i, i)     =  1.0; bin(mcut + 2*i)     = tinytype(s.ulim);
            Ain(mcut + 2*i + 1, i) = -1.0; bin(mcut + 2*i + 1) = tinytype(s.ulim);
        }
        if (revision_bounds) {
            for (int k=1; k<N; ++k) for (int a=0; a<NX0; ++a) {
                int row=mcut+2*nw+2*((k-1)*NX0+a);
                Ain.row(row)=s.S.row(k*NX0+a);
                bin(row)=s.state_hi(a,k)-zf(k*NX0+a);
                Ain.row(row+1)=-s.S.row(k*NX0+a);
                bin(row+1)=zf(k*NX0+a)-s.state_lo(a,k);
            }
        }
        int ns = d4n_qp(s, gc, Ain, bin, w);
        o.outer++; o.iters_total += std::abs(ns);
        if (std::getenv("TINYSDP_REVISION") && ns < 0) {
            o.feasible = false;
            std::cout << "[REV-QP-FAIL] pivots=" << -ns << "\n";
            return o;
        }

        Vec z = zf + s.S * w;
        double dz = 0.0;
        for (int k = 0; k < N; ++k) (*states)[k] = z.segment(k*NX0, NX0);
        for (int k = 0; k < N - 1; ++k) (*inputs)[k] = w.segment(k*NU0, NU0);
        for (int k = 0; k < N; ++k) {
            dz = std::max(dz, static_cast<double>(((*states)[k] - Z[k]).cwiseAbs().maxCoeff()));
            Z[k] = (1.0 - alpha) * Z[k] + alpha * (*states)[k];
        }
        std::cout << "[D4N-OUTER] it=" << it << " kkt_solves=" << ns << " dz=" << dz << "\n";
        if (dz < 1e-4) break;
    }
    return o;
}

static D4Out d4_plan(TinySolver* s, const Vec& x_seed, const Mat& Xref6,
                     const std::vector<std::vector<std::array<tinytype,4>>>& sph_cuts,
                     int rows_per_stage, const Mat& Ad, const Mat& Bd,
                     const Mat& xlo0, const Mat& xhi0, const Mat& ulo, const Mat& uhi,
                     std::vector<Vec>* states, std::vector<Vec>* inputs)
{
    D4Out o;
    tiny_set_x_ref(s, Xref6);
    tiny_set_u_ref(s, Mat::Zero(NU0, N - 1));
    tiny_set_x0(s, x_seed);

    // Linearise about the guide reference on the first pass.
    std::vector<Vec> Z(N, Vec::Zero(NX0));
    for (int k = 0; k < N; ++k) Z[k] = Xref6.col(k);
    Z[0] = x_seed;

    states->assign(N, Vec::Zero(NX0));
    inputs->assign(N - 1, Vec::Zero(NU0));

    const char* mo = std::getenv("TINYSDP_D4_OUTER");
    const int max_outer = mo ? atoi(mo) : 8;
    const double dz_tol = 1e-3;

    // A linearised ball constraint with no trust region oscillates: each new
    // normal flings the path to the other side of the obstacle.  Bound the step
    // to a shrinking box about the current iterate -- standard SCP hygiene, and
    // it is free here because the solver already carries per-stage state boxes.
    const char* trs = std::getenv("TINYSDP_D4_TRUST");
    double trust = trs ? atof(trs) : 1.5;
    const double trust_shrink = 0.75;

    for (int it = 0; it < max_outer; ++it) {
        d4_set_cuts(s, rows_per_stage, sph_cuts, Z);
        Mat xlo = xlo0, xhi = xhi0;
        for (int k = 1; k < N; ++k) {
            for (int i = 0; i < 3; ++i) {
                xlo(i, k) = std::max(xlo0(i, k), tinytype(Z[k](i) - trust));
                xhi(i, k) = std::min(xhi0(i, k), tinytype(Z[k](i) + trust));
            }
        }
        tiny_set_bound_constraints(s, xlo, xhi, ulo, uhi);
        tiny_solve(s);
        o.outer++;
        o.iters_total += s->solution->iter;

        // Roll the returned inputs forward: this is the trajectory that would
        // actually be executed, and it is what we linearise about next.
        Vec x = x_seed;
        (*states)[0] = x;
        double dz = 0.0;
        for (int k = 0; k < N - 1; ++k) {
            Vec u = s->solution->u.col(k).topRows(NU0);
            (*inputs)[k] = u;
            x = Ad * x + Bd * u;
            (*states)[k + 1] = x;
        }
        // Damped update of the linearisation point.  Taking the full step makes
        // the outer loop limit-cycle: each new normal flings the path across
        // the obstacle and the next one flings it back (measured: dz plateaus
        // near 1 m and never falls).
        const char* as = std::getenv("TINYSDP_D4_ALPHA");
        const double alpha = as ? atof(as) : 0.5;
        for (int k = 0; k < N; ++k) {
            dz = std::max(dz, static_cast<double>(((*states)[k] - Z[k]).cwiseAbs().maxCoeff()));
            Z[k] = (1.0 - alpha) * Z[k] + alpha * (*states)[k];
        }
        std::cout << "[D4-OUTER] it=" << it << " admm_iter=" << s->solution->iter
                  << " solved=" << s->solution->solved << " dz=" << dz
                  << " pri_x=" << s->work->primal_residual_state
                  << " pri_u=" << s->work->primal_residual_input
                  << " dua_x=" << s->work->dual_residual_state
                  << " dua_u=" << s->work->dual_residual_input << "\n";
        if (dz < dz_tol) break;
        trust *= trust_shrink;
    }
    return o;
}
// ========================== END D4 ==========================

void destroy_solver(TinySolver* solver) {

    if (!solver) return;
    delete solver->solution;
    delete solver->settings;
    delete solver->cache;
    delete solver->work;
    delete solver;
}

std::filesystem::path resolve_output_dir() {
    namespace fs = std::filesystem;
    const char* raw = std::getenv("TINYSDP_OUTPUT_DIR");
    fs::path out = (raw && raw[0]) ? fs::path(raw) : fs::current_path() / "outputs";
    std::error_code ec;
    fs::create_directories(out, ec);
    if (ec) {
        std::cout << "[TinySDP-3D] Could not create output directory " << out
                  << ": " << ec.message() << "\n";
    }
    return out;
}

int getenv_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    try {
        return std::max(1, std::stoi(raw));
    } catch (...) {
        return fallback;
    }
}

bool getenv_flag(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
    if (value == "0" || value == "false" || value == "no" || value == "off") return false;
    return fallback;
}

std::string getenv_string(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return fallback;
    return std::string(raw);
}

std::vector<std::string> getenv_list(const char* name) {
    std::vector<std::string> values;
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return values;
    std::string all(raw);
    std::size_t start = 0;
    while (start < all.size()) {
        std::size_t comma = all.find(',', start);
        std::string token = all.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(token);
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return values;
}

RunConfig load_run_config() {
    RunConfig cfg;
    cfg.replan_stride = getenv_int("TINYSDP_3D_REPLAN_STRIDE", N - 1);
    cfg.file_prefix = getenv_string("TINYSDP_3D_FILE_PREFIX", "tinysdp_3d_");
    cfg.log_plan_states = getenv_flag("TINYSDP_3D_LOG_PLAN_STATES", false);
    cfg.mocap_mode = getenv_flag("TINYSDP_3D_MOCAP_MODE", false);
    cfg.scenario_filter = getenv_list("TINYSDP_3D_SCENARIOS");
    if (cfg.scenario_filter.empty()) {
        cfg.scenario_filter.push_back(TINYSDP_SCENARIO_SLUG);
    }
    return cfg;
}

bool scenario_selected(const RunConfig& cfg, const std::string& slug) {
    if (cfg.scenario_filter.empty()) return true;
    return std::find(cfg.scenario_filter.begin(), cfg.scenario_filter.end(), slug) != cfg.scenario_filter.end();
}

std::vector<ScenarioSpec> build_scenarios() {
    std::vector<ScenarioSpec> scenarios;

    ScenarioSpec sweeper;
    sweeper.name = "Sweeping Barrier";
    sweeper.slug = "sweeping_barrier";
    sweeper.prediction_inflation = tinytype(0.016);
    sweeper.activation_on = 2.0;
    sweeper.activation_off = 2.25;
    sweeper.total_steps = 28;
    sweeper.guide_points = {
        Eigen::Vector3d(-3.0, -1.35, 0.95),
        Eigen::Vector3d(-1.8, -1.75, 1.10),
        Eigen::Vector3d(-0.8, -0.65, 0.55),
    };
    sweeper.obstacles.agents = {
        { tinytype(-3.3), tinytype( 0.0), tinytype(0.20), tinytype(0.00), tinytype( 0.00), tinytype(0.00),
          tinytype(0.65), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0) },
        { tinytype(-1.8), tinytype( 1.30), tinytype(0.35), tinytype(0.03), tinytype(-0.16), tinytype(0.00),
          tinytype(0.56), tinytype(0.03), tinytype(0.30), tinytype(0.2), tinytype(0.05), tinytype(0.30), tinytype(0.4), tinytype(0.04), tinytype(0.40), tinytype(0.5) },
        { tinytype(-1.35), tinytype( 1.55), tinytype(0.95), tinytype(0.03), tinytype(-0.16), tinytype(0.00),
          tinytype(0.55), tinytype(0.03), tinytype(0.30), tinytype(0.8), tinytype(0.05), tinytype(0.30), tinytype(0.7), tinytype(0.04), tinytype(0.40), tinytype(1.0) }
    };
    scenarios.push_back(sweeper);

    // ADDED for a like-for-like comparison against a continuous-curve planner
    // (not part of upstream TinySDP).  Identical to "sweeping_barrier" in
    // start, goal, guide points and sphere centres/radii at t = 0, with every
    // sphere velocity and wobble amplitude set to zero, so both planners face
    // exactly the same STATIC scene.
    ScenarioSpec frozen = sweeper;
    frozen.name = "Frozen Barrier";
    frozen.slug = "frozen_barrier";
    for (auto& a : frozen.obstacles.agents) {
        a.vx = a.vy = a.vz = tinytype(0);
        a.wobble_x = a.wobble_y = a.wobble_z = tinytype(0);
    }
    scenarios.push_back(frozen);

    // ADDED (not upstream): combinatorially hard static scenes.  Spheres sit ON
    // the straight chord from the start to the goal, so each one is a genuine
    // left/right decision rather than a nudge.  Guide points route around them
    // with 0.35 m to spare, so the route hint HELPS this planner rather than
    // handicapping it.
    {
        struct HardSpec { const char* slug; const char* name; int nsph; double r;
                          double ts[3]; };
        const HardSpec specs[] = {
            {"chord1", "Chord 1", 1, 0.40, {0.50, 0.0,      0.0}},
            {"chord2", "Chord 2", 2, 0.40, {1.0/3.0, 2.0/3.0, 0.0}},
            {"chord3", "Chord 3", 3, 0.35, {0.25, 0.50,    0.75}},
        };
        const Eigen::Vector3d p_start(-4.5, 0.0, 1.0), p_goal(0.0, 0.0, 0.0);
        const Eigen::Vector3d dir = p_goal - p_start;
        for (const auto& sp : specs) {
            ScenarioSpec sc;
            sc.name = sp.name; sc.slug = sp.slug;
            sc.prediction_inflation = tinytype(0.0);
            sc.activation_on = 2.0; sc.activation_off = 2.25;
            sc.total_steps = 28;
            sc.obstacles.agents.clear();
            sc.guide_points.clear();
            for (int i = 0; i < sp.nsph; ++i) {
                Eigen::Vector3d c = p_start + sp.ts[i] * dir;
                sc.obstacles.agents.push_back(
                    { tinytype(c.x()), tinytype(c.y()), tinytype(c.z()),
                      tinytype(0), tinytype(0), tinytype(0), tinytype(sp.r),
                      tinytype(0), tinytype(0), tinytype(0),
                      tinytype(0), tinytype(0), tinytype(0),
                      tinytype(0), tinytype(0), tinytype(0) });
                sc.guide_points.push_back(
                    Eigen::Vector3d(c.x(), c.y() - (sp.r + 0.35), c.z()));
            }
            scenarios.push_back(sc);
        }
    }

    // ADDED (not upstream): an ASYMMETRIC scene.  Two balls straddle the chord
    // on opposite sides, so the cheap route weaves and a single passing side
    // cannot express it.  Guide points follow the weave, so the route hint
    // helps this planner rather than handicapping it.
    {
        ScenarioSpec sc;
        sc.name = "Weave 2"; sc.slug = "weave2";
        sc.prediction_inflation = tinytype(0.0);
        sc.activation_on = 2.0; sc.activation_off = 2.25;
        sc.total_steps = 28;
        const Eigen::Vector3d p_start(-4.5, 0.0, 1.0), p_goal(0.0, 0.0, 0.0);
        const Eigen::Vector3d dir = p_goal - p_start;
        const Eigen::Vector3d ex = dir.normalized();
        Eigen::Vector3d e1 = ex.cross(Eigen::Vector3d(0, 0, 1)).normalized();
        const double ts[2] = {0.35, 0.65};
        const double sgn[2] = {+1.0, -1.0};
        for (int i = 0; i < 2; ++i) {
            Eigen::Vector3d c = p_start + ts[i] * dir + sgn[i] * 0.25 * e1;
            sc.obstacles.agents.push_back(
                { tinytype(c.x()), tinytype(c.y()), tinytype(c.z()),
                  tinytype(0), tinytype(0), tinytype(0), tinytype(0.40),
                  tinytype(0), tinytype(0), tinytype(0),
                  tinytype(0), tinytype(0), tinytype(0),
                  tinytype(0), tinytype(0), tinytype(0) });
            Eigen::Vector3d g = c - sgn[i] * 0.75 * e1;
            sc.guide_points.push_back(g);
        }
        scenarios.push_back(sc);
    }

    ScenarioSpec gate;
    gate.name = "Rising Gate";
    gate.slug = "vertical_gate";
    gate.prediction_inflation = tinytype(0.015);
    gate.activation_on = 1.45;
    gate.activation_off = 1.70;
    gate.total_steps = 26;
    gate.guide_points = {
        Eigen::Vector3d(-3.1,  0.0, 2.45),
        Eigen::Vector3d(-1.6,  0.0, 2.85),
        Eigen::Vector3d(-0.5,  0.0, 1.00),
    };
    gate.obstacles.agents = {
        { tinytype(-2.9), tinytype( 0.90), tinytype(0.15), tinytype(0.00), tinytype(0.00), tinytype(0.00),
          tinytype(0.60), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0) },
        { tinytype(-2.4), tinytype(-0.90), tinytype(0.15), tinytype(0.00), tinytype(0.00), tinytype(0.00),
          tinytype(0.60), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0) },
        { tinytype(-1.35), tinytype(0.00), tinytype(0.35), tinytype(0.00), tinytype(0.00), tinytype(0.00),
          tinytype(0.55), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.00), tinytype(0.00), tinytype(0.0), tinytype(0.75), tinytype(0.48), tinytype(-1.1) }
    };
    scenarios.push_back(gate);

    return scenarios;
}

ScenarioResult run_scenario(const ScenarioSpec& scenario,
                            const RunConfig& cfg,
                            const std::filesystem::path& output_dir,
                            const Mat& Ad,
                            const Mat& Bd,
                            const Mat& A,
                            const Mat& B,
                            const Mat& Q,
                            const Mat& R,
                            const Mat& x_min,
                            const Mat& x_max,
                            const Mat& u_min,
                            const Mat& u_max,
                            const Vec& fdyn,
                            tinytype rho_base,
                            tinytype rho_psd_penalty) {
    ScenarioResult result;
    result.name = scenario.name;
    result.slug = scenario.slug;

    const int nxL = A.rows();
    const int nuL = B.cols();
    const tinytype goal_pos_tol = tinytype(0.15);
    const tinytype goal_vel_tol = tinytype(0.05);
    const tinytype tracker_input_limit = tinytype(3.0);
    const double seg_guard = 0.02;
    const int replan_stride = cfg.replan_stride;
    const int horizon_guard = 1;
    bool psd_constraints_active = !cfg.mocap_mode;
    const int total_steps = cfg.mocap_mode ? (scenario.total_steps + 12) : scenario.total_steps;
    const tinytype terminal_capture_radius = cfg.mocap_mode ? tinytype(3.0) : tinytype(1.0);

    Vec x0(NX0);
    x0 << -4.5, 0.0, 1.0, 0.0, 0.0, 0.0;
    Vec x_track = x0;
    PlanCache plan;

    auto goal_reached = [&](const Vec& state) -> bool {
        return state.topRows(3).norm() < goal_pos_tol &&
               state.bottomRows(3).norm() < goal_vel_tol;
    };

    TinySolver* solver_psd = nullptr;
    if (tiny_setup(&solver_psd, A, B, fdyn, Q, R, rho_base, nxL, nuL, N, /*verbose=*/0)) {
        std::cout << "[TinySDP-3D] Failed to setup planner for " << scenario.slug << "\n";
        return result;
    }
    solver_psd->settings->adaptive_rho = 0;
    solver_psd->settings->max_iter = getenv_int("TINYSDP_MAXITER", 1200);
    solver_psd->settings->abs_pri_tol = tinytype(5e-2);
    solver_psd->settings->abs_dua_tol = tinytype(5e-2);
    solver_psd->settings->check_termination = 1;
    { const char* tt = std::getenv("TINYSDP_TOL"); if (tt) { solver_psd->settings->abs_pri_tol = tinytype(atof(tt)); solver_psd->settings->abs_dua_tol = tinytype(atof(tt)); } }
    tiny_set_bound_constraints(solver_psd, x_min, x_max, u_min, u_max);
    tiny_enable_psd(solver_psd, NX0, NU0, rho_psd_penalty);
    tiny_set_x_ref(solver_psd, Mat::Zero(nxL, N));
    tiny_set_u_ref(solver_psd, Mat::Zero(nuL, N - 1));

    TinySolver* solver_track = nullptr;
    Mat Q_track = Mat::Zero(NX0, NX0);
    Q_track(0,0) = 55.0; Q_track(1,1) = 55.0; Q_track(2,2) = 55.0;
    Q_track(3,3) = 6.0;  Q_track(4,4) = 6.0;  Q_track(5,5) = 6.0;
    Mat R_track = Mat::Zero(NU0, NU0);
    R_track.diagonal().array() = tinytype(0.25);
    Vec fdyn_track = Vec::Zero(NX0);
    Mat x_min_track = Mat::Constant(NX0, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max_track = Mat::Constant(NX0, N,  std::numeric_limits<tinytype>::infinity());
    x_min_track.topRows(NX0).setConstant(-30.0);
    x_max_track.topRows(NX0).setConstant( 30.0);
    Mat u_min_track = Mat::Constant(NU0, N - 1, -3.0);
    Mat u_max_track = Mat::Constant(NU0, N - 1,  3.0);
    if (tiny_setup(&solver_track, Ad, Bd, fdyn_track, Q_track, R_track, rho_base, NX0, NU0, N, /*verbose=*/0)) {
        destroy_solver(solver_psd);
        std::cout << "[TinySDP-3D] Failed to setup tracker for " << scenario.slug << "\n";
        return result;
    }
    solver_track->settings->adaptive_rho = 0;
    solver_track->settings->max_iter = 120;
    solver_track->settings->abs_pri_tol = tinytype(1e-3);
    solver_track->settings->abs_dua_tol = tinytype(1e-3);
    solver_track->settings->check_termination = 1;
    tiny_set_bound_constraints(solver_track, x_min_track, x_max_track, u_min_track, u_max_track);
    tiny_set_x_ref(solver_track, Mat::Zero(NX0, N));
    tiny_set_u_ref(solver_track, Mat::Zero(NU0, N - 1));

    // D4 planner: the same problem the lifted planner is trying to solve, but
    // with rank one imposed, so it lives in the base 6/3 space with no cone.
    TinySolver* solver_d4 = nullptr;
    D4N d4n;
    Mat d4_xlo, d4_xhi, d4_ulo, d4_uhi;
    const bool use_d4 = (std::getenv("TINYSDP_D4") != nullptr);
    if (use_d4) {
        Mat Q_d4 = Q.topLeftCorner(NX0, NX0);
        Mat R_d4 = R.topLeftCorner(NU0, NU0);
        Mat x_min_d4 = x_min.topRows(NX0);
        Mat x_max_d4 = x_max.topRows(NX0);
        Mat u_min_d4 = u_min.topRows(NU0);
        Mat u_max_d4 = u_max.topRows(NU0);
        const char* rs = std::getenv("TINYSDP_D4_RHO");
        const tinytype rho_d4 = rs ? tinytype(atof(rs)) : rho_base;
        if (tiny_setup(&solver_d4, Ad, Bd, fdyn_track, Q_d4, R_d4, rho_d4, NX0, NU0, N, 0)) {
            destroy_solver(solver_psd); destroy_solver(solver_track);
            return result;
        }
        solver_d4->settings->adaptive_rho = 0;
        { const char* mi = std::getenv("TINYSDP_D4_MAXITER"); solver_d4->settings->max_iter = mi ? atoi(mi) : 500; }
        { const char* dt = std::getenv("TINYSDP_D4_TOL"); double v = dt ? atof(dt) : 1e-3;
          solver_d4->settings->abs_pri_tol = tinytype(v);
          solver_d4->settings->abs_dua_tol = tinytype(v); }
        solver_d4->settings->check_termination = 1;
        tiny_set_bound_constraints(solver_d4, x_min_d4, x_max_d4, u_min_d4, u_max_d4);
        tiny_enable_per_stage_state_linear(solver_d4, 8);
        d4_xlo = x_min_d4; d4_xhi = x_max_d4; d4_ulo = u_min_d4; d4_uhi = u_max_d4;
        if (std::getenv("TINYSDP_D4_NEWTON")) {
            d4n_build(d4n, Ad, Bd, Q_d4, R_d4, 3.0);
            d4n.state_lo=x_min_d4; d4n.state_hi=x_max_d4;
        }
    }

    std::ofstream csv_track(output_dir / (cfg.file_prefix + scenario.slug + "_tracking.csv"));
    if (csv_track.is_open()) {
        csv_track << "k,x,y,z,vx,vy,vz,u1,u2,u3,signed_dist,seg_signed_dist,plan_age,solver_iter\n";
    }
    std::ofstream csv_spheres(output_dir / (cfg.file_prefix + scenario.slug + "_spheres.csv"));
    if (csv_spheres.is_open()) {
        csv_spheres << "k,sphere,cx,cy,cz,r\n";
    }
    std::ofstream csv_plan(output_dir / (cfg.file_prefix + scenario.slug + "_plan_log.csv"));
    if (csv_plan.is_open()) {
        csv_plan << "replan_step,plan_type,iter,num_spheres,min_sd_seed,threshold_on,threshold_off,goal_dist,certified_future,status\n";
    }
    std::ofstream csv_cert(output_dir / (cfg.file_prefix + scenario.slug + "_certificate.csv"));
    if (csv_cert.is_open()) {
        csv_cert << "k,trace_gap,eta_min,true_dist2_min,certified\n";
    }
    std::ofstream csv_plan_states;
    if (cfg.log_plan_states) {
        csv_plan_states.open(output_dir / (cfg.file_prefix + scenario.slug + "_plan_states.csv"));
        if (csv_plan_states.is_open()) {
            csv_plan_states << "replan_step,h,x,y,z,vx,vy,vz\n";
        }
    }

    auto log_spheres = [&](int step, const std::vector<std::array<tinytype,4>>& spheres_now) {
        if (!csv_spheres.is_open()) return;
        for (std::size_t j = 0; j < spheres_now.size(); ++j) {
            csv_spheres << step << "," << j
                       << "," << spheres_now[j][0]
                       << "," << spheres_now[j][1]
                       << "," << spheres_now[j][2]
                       << "," << spheres_now[j][3] << "\n";
        }
    };

    auto log_tracking_row = [&](int step, const Vec& state, const Vec& input,
                                double sd_point, double sd_segment,
                                int plan_age, int iters) {
        if (!csv_track.is_open()) return;
        csv_track << step
                  << "," << state(0) << "," << state(1) << "," << state(2)
                  << "," << state(3) << "," << state(4) << "," << state(5)
                  << "," << input(0) << "," << input(1) << "," << input(2)
                  << "," << sd_point << "," << sd_segment
                  << "," << plan_age << "," << iters << "\n";
    };

    auto log_certificate = [&](int step,
                               const Vec& x_lifted,
                               const std::vector<std::array<tinytype,4>>& spheres_now) {
        PsdCertificate cert = compute_psd_certificate(x_lifted, spheres_now);
        result.min_eta = std::min(result.min_eta, cert.eta_min);
        result.max_trace_gap = std::max(result.max_trace_gap, std::abs(cert.trace_gap));
        result.all_certified = result.all_certified && cert.certified;
        if (!csv_cert.is_open()) return;
        csv_cert << step << "," << cert.trace_gap << "," << cert.eta_min << ","
                 << cert.true_dist2_min << "," << (cert.certified ? 1 : 0) << "\n";
    };

    auto choose_safe_input = [&](const Vec& current_state,
                                 const std::vector<Vec>& candidates,
                                 const std::vector<std::array<tinytype,4>>& next_spheres,
                                 const Vec& target_state,
                                 Vec* u_safe) -> bool {
        static const std::array<double, 7> scales = {1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15};
        bool found = false;
        double best_score = std::numeric_limits<double>::infinity();
        Vec best_u = Vec::Zero(NU0);

        for (const Vec& u_nominal : candidates) {
            for (double scale : scales) {
                Vec u_try = clamp_input(tinytype(scale) * u_nominal, tracker_input_limit);
                Vec x1_try = Ad * current_state + Bd * u_try;
                PsdCertificate cert_try = compute_psd_certificate(build_lifted(x1_try), next_spheres);
                double seg_try = signed_distance_segment_spheres(current_state, x1_try, next_spheres);
                if (!cert_try.certified || seg_try < seg_guard) {
                    continue;
                }

                double goal_score = x1_try.topRows(3).norm();
                double plan_score = (x1_try - target_state).norm();
                double effort_score = u_try.norm();
                double score = goal_score + 0.55 * plan_score + 0.05 * effort_score;
                if (!found || score < best_score) {
                    found = true;
                    best_score = score;
                    best_u = u_try;
                }
            }
        }

        if (!found) {
            return false;
        }
        *u_safe = best_u;
        return true;
    };

    const bool revision = std::getenv("TINYSDP_REVISION") != nullptr;
    // Dense interstage diagnostic on the modeled constant-acceleration path
    // and all continuously moving balls. This is a sampled diagnostic, not
    // a mathematical safety certificate between its samples.
    auto dense_clearance = [&](const Vec& x, const Vec& u, int absolute_step) {
        double smallest = std::numeric_limits<double>::infinity();
        const double dt = double(Ad(0,3));
        for (int q = 0; q <= 64; ++q) {
            double tau = dt * double(q) / 64.0;
            Eigen::Vector3d point = x.head(3).cast<double>() +
                tau * x.tail(3).cast<double>() + 0.5 * tau * tau * u.cast<double>();
            for (const auto& obstacle : scenario.obstacles.agents) {
                auto ball = obstacle.sphere_at_time(tinytype((absolute_step + double(q)/64.0) * double(scenario.obstacles.dt)));
                Eigen::Vector3d c{double(ball[0]), double(ball[1]), double(ball[2])};
                smallest = std::min(smallest, (point-c).norm()-double(ball[3]));
            }
        }
        return smallest;
    };
    auto replan_psd = [&](int step, const Vec& x_seed) {
        const auto revision_start = std::chrono::steady_clock::now();
        auto spheres_now_all = scenario.obstacles.spheres_at_step(step);
        std::vector<std::array<tinytype, 4>> spheres_now_static;
        std::vector<std::array<tinytype, 4>> spheres_now_dynamic;
        spheres_now_static.reserve(spheres_now_all.size());
        spheres_now_dynamic.reserve(spheres_now_all.size());
        for (std::size_t i = 0; i < spheres_now_all.size(); ++i) {
            if (i < scenario.obstacles.agents.size() && sphere_has_motion(scenario.obstacles.agents[i])) {
                spheres_now_dynamic.push_back(spheres_now_all[i]);
            } else {
                spheres_now_static.push_back(spheres_now_all[i]);
            }
        }

        double sd_seed = signed_distance_point_spheres(x_seed, spheres_now_all);
        double sd_seed_dynamic = spheres_now_dynamic.empty()
            ? std::numeric_limits<double>::infinity()
            : signed_distance_point_spheres(x_seed, spheres_now_dynamic);
        double goal_dist = x_seed.topRows(3).norm();

        double on_thresh = scenario.activation_on;
        double off_thresh = scenario.activation_off;
        if (cfg.mocap_mode) {
            on_thresh = std::min(on_thresh, goal_dist + 0.30);
            off_thresh = std::max(on_thresh + 0.20, std::min(off_thresh, goal_dist + 0.70));
            if (!psd_constraints_active && sd_seed_dynamic < on_thresh) {
                psd_constraints_active = true;
            } else if (psd_constraints_active && sd_seed_dynamic > off_thresh) {
                psd_constraints_active = false;
            }
        }

        std::vector<std::array<tinytype, 4>> planner_spheres_now = spheres_now_static;
        if (!cfg.mocap_mode || psd_constraints_active) {
            planner_spheres_now.insert(planner_spheres_now.end(),
                                       spheres_now_dynamic.begin(),
                                       spheres_now_dynamic.end());
        }

        std::vector<std::vector<std::array<tinytype, 4>>> predicted_true;
        std::vector<std::vector<std::array<tinytype, 4>>> predicted_inflated;
        predicted_true.reserve(N);
        predicted_inflated.reserve(N);
        if (cfg.mocap_mode) {
            for (int h = 0; h < N; ++h) {
                predicted_true.push_back(planner_spheres_now);
                predicted_inflated.push_back(planner_spheres_now);
            }
        } else if (psd_constraints_active) {
            if (cfg.mocap_mode) {
                for (int h = 0; h < N; ++h) {
                    predicted_true.push_back(planner_spheres_now);
                    predicted_inflated.push_back(planner_spheres_now);
                }
            } else {
                predicted_true = scenario.obstacles.horizon_spheres_per_stage(step, N, tinytype(0.0));
                predicted_inflated =
                    scenario.obstacles.horizon_spheres_per_stage(step, N, scenario.prediction_inflation);
            }
        } else {
            for (int h = 0; h < N; ++h) {
                predicted_true.emplace_back();
                predicted_inflated.emplace_back();
            }
        }

        Mat Xref_plan = Mat::Zero(nxL, N);
        Mat Uref_plan = Mat::Zero(nuL, N - 1);
        std::vector<Eigen::Vector3d> route;
        route.reserve(scenario.guide_points.size() + 2);
        route.push_back(Eigen::Vector3d(
            static_cast<double>(x_seed(0)),
            static_cast<double>(x_seed(1)),
            static_cast<double>(x_seed(2))));
        if (!cfg.mocap_mode || psd_constraints_active) {
            for (const auto& p : scenario.guide_points) {
                if (p.x() > static_cast<double>(x_seed(0)) + 0.05) {
                    route.push_back(p);
                }
            }
        }
        route.push_back(Eigen::Vector3d::Zero());
        for (int h = 0; h < N; ++h) {
            double alpha = (N > 1) ? static_cast<double>(h) / static_cast<double>(N - 1) : 1.0;
            Eigen::Vector3d pref = sample_polyline(route, alpha);
            Xref_plan(0, h) = tinytype(pref.x());
            Xref_plan(1, h) = tinytype(pref.y());
            Xref_plan(2, h) = tinytype(pref.z());
            Xref_plan(3, h) = tinytype(0.0);
            Xref_plan(4, h) = tinytype(0.0);
            Xref_plan(5, h) = tinytype(0.0);
        }
        tiny_set_x_ref(solver_psd, Xref_plan);
        tiny_set_u_ref(solver_psd, Uref_plan);
        solver_psd->settings->en_psd = planner_spheres_now.empty() ? 0 : 1;
        if (std::getenv("TINYSDP_ABLATE_PSD")) solver_psd->settings->en_psd = 0;
        if (!planner_spheres_now.empty()) {
            if (cfg.mocap_mode) {
                tiny_set_lifted_spheres(solver_psd, planner_spheres_now);
            } else {
                tiny_set_lifted_spheres_per_stage(solver_psd, predicted_inflated);
            }
        }
        tiny_set_x0(solver_psd, build_lifted(x_seed));

        std::vector<Vec> d4_states, d4_inputs;
        D4Out d4o;
        int d4_rows = 0;
        for (const auto& v : predicted_inflated) d4_rows = std::max<int>(d4_rows, (int)v.size());
        d4_rows = std::min(d4_rows, 8);
        if (std::getenv("TINYSDP_D4_NOCUT")) d4_rows = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        if (use_d4) {
            if (d4n.ready) {
                d4o = d4_plan_newton(d4n, x_seed, Xref_plan.topRows(NX0),
                                     predicted_inflated, &d4_states, &d4_inputs);
            } else {
                d4o = d4_plan(solver_d4, x_seed, Xref_plan.topRows(NX0),
                              predicted_inflated, d4_rows, Ad, Bd,
                              d4_xlo, d4_xhi, d4_ulo, d4_uhi, &d4_states, &d4_inputs);
            }
        } else {
            tiny_solve(solver_psd);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        result.planner_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        result.planner_solves++;

        bool plan_valid = true;
        int certified_future = 0;
        int first_fail = -1;
        if (use_d4) {
            // Nothing to certify away: the moment matrix IS the outer product,
            // so the relaxed distance and the true distance are the same number.
            for (int h = 1; h < N - 1; ++h) {
                double sd = predicted_true[h].empty()
                    ? std::numeric_limits<double>::infinity()
                    : signed_distance_point_spheres(d4_states[h], predicted_true[h]);
                std::cout << "[CERT-D4] h=" << h << " clearance=" << sd
                          << " certified=" << (sd >= 0.0 ? 1 : 0) << "\n";
                if (sd >= 0.0) { certified_future++; }
                else { plan_valid = false; if (first_fail < 0) first_fail = h; break; }
            }
            std::cout << "[D4] outer=" << d4o.outer << " admm_iters=" << d4o.iters_total << "\n";
        } else
        for (int h = 1; h < N - 1; ++h) {
            PsdCertificate cert_h = compute_psd_certificate(solver_psd->solution->x.col(h), predicted_true[h]);
            std::cout << "[CERT-SOLVER] h=" << h
                      << " trace_gap=" << cert_h.trace_gap
                      << " eta_min=" << cert_h.eta_min
                      << " true_dist2_min=" << cert_h.true_dist2_min
                      << " certified=" << (cert_h.certified?1:0) << "\n";
            if (cert_h.certified) {
                certified_future++;
            } else {
                plan_valid = false;
                if (first_fail < 0) first_fail = h;
                break;
            }
        }

        double revision_stage = std::numeric_limits<double>::infinity();
        double revision_dense = std::numeric_limits<double>::infinity();
        if (revision) {
            PlanCache candidate;
            // Use the actual dynamics rollout that would be passed to tracker.
            if (use_d4) { candidate.states = d4_states; candidate.inputs = d4_inputs; }
            else { rollout_plan(Ad, Bd, x_seed, solver_psd, &candidate); }
            if (use_d4 && !d4o.feasible) plan_valid = false;
            for (int h = 0; h < N; ++h) {
                double sd = signed_distance_point_spheres(candidate.states[h], predicted_true[h]);
                revision_stage = std::min(revision_stage, sd);
                if (!std::isfinite(sd) || sd < -1e-9) plan_valid = false;
                if (h < N-1) {
                    double dense = dense_clearance(candidate.states[h], candidate.inputs[h], step+h);
                    revision_dense = std::min(revision_dense, dense);
                    if (!std::isfinite(dense) || dense < -1e-9) plan_valid = false;
                }
            }
            const double full_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-revision_start).count();
            const double solve_us = std::chrono::duration<double, std::micro>(t1-t0).count();
            std::cout << "[REV-PLAN] step=" << step << " solve_us=" << solve_us
                      << " total_us=" << full_us << " stage_clear=" << revision_stage
                      << " dense_clear=" << revision_dense << " valid=" << plan_valid
                      << " qp_ok=" << (!use_d4 || d4o.feasible) << "\n";
        }
        if (plan_valid) {
            if (use_d4) {
                plan.states = d4_states; plan.inputs = d4_inputs;
                plan.last_iters = d4o.iters_total;
            } else {
                rollout_plan(Ad, Bd, x_seed, solver_psd, &plan);
            }
            plan.start_step = step;
            for (size_t hh = 0; hh < plan.states.size(); ++hh) {
                std::cout << "[PLANSTATE] " << hh
                          << " " << plan.states[hh](0)
                          << " " << plan.states[hh](1)
                          << " " << plan.states[hh](2) << "\n";
            }
            if (csv_plan_states.is_open()) {
                for (int h = 0; h < static_cast<int>(plan.states.size()); ++h) {
                    const Vec& xh = plan.states[h];
                    csv_plan_states << step << "," << h
                                    << "," << xh(0) << "," << xh(1) << "," << xh(2)
                                    << "," << xh(3) << "," << xh(4) << "," << xh(5) << "\n";
                }
            }
        } else {
            std::cout << "[TinySDP-3D][" << scenario.slug << "] Planner reject at k=" << step
                      << " first_fail_stage=" << first_fail << "\n";
        }

        if (csv_plan.is_open()) {
            csv_plan << step << "," << (cfg.mocap_mode
                                            ? (psd_constraints_active ? "tinysdp_current" : "static_only")
                                            : "tinysdp_horizon")
                     << "," << solver_psd->solution->iter
                     << "," << planner_spheres_now.size()
                     << "," << sd_seed
                     << "," << on_thresh
                     << "," << off_thresh
                     << "," << goal_dist
                     << "," << certified_future
                     << "," << (plan_valid ? "accepted" : "rejected") << "\n";
        }
    };

    auto spheres0 = scenario.obstacles.spheres_at_step(0);
    log_spheres(0, spheres0);
    double sd0 = signed_distance_point_spheres(x_track, spheres0);
    Vec zero_u = Vec::Zero(NU0);
    log_tracking_row(0, x_track, zero_u, sd0, sd0, 0, 0);
    log_certificate(0, build_lifted(x_track), spheres0);
    result.min_point_sd = sd0;
    result.min_seg_sd = sd0;

    replan_psd(0, x_track);

    Vec prev_state = x_track;
    for (int k = 0; k < total_steps; ++k) {
        const auto revision_step_start = std::chrono::steady_clock::now();
        bool terminal_capture = x_track.topRows(3).norm() < terminal_capture_radius;
        bool need_replan = !terminal_capture &&
                           ((k == 0) ||
                            (k - plan.start_step >= replan_stride) ||
                            (k >= plan.start_step + N - horizon_guard));
        if (need_replan && k > 0) {
            replan_psd(k, x_track);
        }

        auto next_spheres = scenario.obstacles.spheres_at_step(k + 1);
        Vec u0 = Vec::Zero(NU0);
        int applied_iters = 0;

        if (terminal_capture) {
            Vec u_fb(NU0);
            u_fb = -tinytype(0.9) * x_track.topRows(3) - tinytype(1.6) * x_track.bottomRows(3);
            u_fb = clamp_input(u_fb, tracker_input_limit);
            if (!choose_safe_input(x_track, {u_fb}, next_spheres, Vec::Zero(NX0), &u0)) {
                u0.setZero();
            }
        } else if (!plan.inputs.empty()) {
            int offset = clamp_index(k - plan.start_step, 0, static_cast<int>(plan.inputs.size()) - 1);
            Vec plan_target = plan.states[clamp_index(offset + 1, 0, static_cast<int>(plan.states.size()) - 1)];
            Vec u_plan = plan.inputs[offset];
            set_base_tracking_refs(solver_track, plan, k);
            tiny_set_x0(solver_track, x_track);
            auto t0 = std::chrono::high_resolution_clock::now();
            tiny_solve(solver_track);
            auto t1 = std::chrono::high_resolution_clock::now();
            result.tracker_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.tracker_solves++;
            applied_iters = solver_track->solution->iter;

            Vec u_nominal = solver_track->solution->u.col(0);
            Vec u_fb(NU0);
            u_fb = -tinytype(0.9) * x_track.topRows(3) - tinytype(1.6) * x_track.bottomRows(3);
            u_fb = clamp_input(u_fb, tracker_input_limit);

            std::vector<Vec> candidates;
            candidates.push_back(u_nominal);
            candidates.push_back(u_plan);
            candidates.push_back(tinytype(0.5) * (u_nominal + u_plan));
            candidates.push_back(u_fb);

            if (!choose_safe_input(x_track, candidates, next_spheres, plan_target, &u0)) {
                Vec u_brake(NU0);
                u_brake = -tinytype(1.6) * x_track.bottomRows(3);
                u_brake = clamp_input(u_brake, tracker_input_limit);
                if (!choose_safe_input(x_track, {u_brake}, next_spheres, plan_target, &u0)) {
                    u0.setZero();
                }
            }
        } else {
            Vec u_brake(NU0);
            u_brake = -tinytype(1.6) * x_track.bottomRows(3);
            u_brake = clamp_input(u_brake, tracker_input_limit);
            if (!choose_safe_input(x_track, {u_brake}, next_spheres, Vec::Zero(NX0), &u0)) {
                u0.setZero();
            }
        }

        if (revision) {
            double dc = dense_clearance(x_track, u0, k);
            double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-revision_step_start).count();
            std::cout << "[REV-STEP] step=" << k << " full_us=" << elapsed << " dense_clear=" << dc << "\n";
        }
        prev_state = x_track;
        x_track = Ad * x_track + Bd * u0;

        int step_idx = k + 1;
        auto spheres_now = scenario.obstacles.spheres_at_step(step_idx);
        log_spheres(step_idx, spheres_now);
        double sd_point = signed_distance_point_spheres(x_track, spheres_now);
        double sd_segment = signed_distance_segment_spheres(prev_state, x_track, spheres_now);
        result.min_point_sd = std::min(result.min_point_sd, sd_point);
        result.min_seg_sd = std::min(result.min_seg_sd, sd_segment);

        int plan_age = step_idx - plan.start_step;
        log_tracking_row(step_idx, x_track, u0, sd_point, sd_segment, plan_age, applied_iters);
        log_certificate(step_idx, build_lifted(x_track), spheres_now);

        if (goal_reached(x_track)) {
            result.success = true;
            result.goal_step = step_idx;
            break;
        }
    }

    result.final_goal_dist = x_track.topRows(3).norm();
    result.final_vel_norm = x_track.bottomRows(3).norm();

    if (solver_d4) destroy_solver(solver_d4);
    destroy_solver(solver_psd);
    destroy_solver(solver_track);
    return result;
}

}  // namespace

int main() {
    const std::filesystem::path output_dir = resolve_output_dir();
    const RunConfig cfg = load_run_config();

    Mat Ad(NX0, NX0);
    Ad << 1, 0, 0, 1, 0, 0,
          0, 1, 0, 0, 1, 0,
          0, 0, 1, 0, 0, 1,
          0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 1;
    Mat Bd(NX0, NU0);
    Bd << 0.5, 0,   0,
          0,   0.5, 0,
          0,   0,   0.5,
          1,   0,   0,
          0,   1,   0,
          0,   0,   1;

    Mat A, B;
    tiny_build_lifted_from_base(Ad, Bd, A, B);
    const int nxL = A.rows();
    const int nuL = B.cols();

    Mat Q = Mat::Zero(nxL, nxL);
    Q(0,0) = 40.0; Q(1,1) = 40.0; Q(2,2) = 40.0;
    Q(3,3) = 4.0;  Q(4,4) = 4.0;  Q(5,5) = 4.0;
    Q.diagonal().segment(NX0, NX0 * NX0).array() = tinytype(1e-4);

    Mat R = Mat::Zero(nuL, nuL);
    const int nxu = NX0 * NU0;
    const int nux = NU0 * NX0;
    const int nuu = NU0 * NU0;
    R.diagonal().head(NU0).array() = tinytype(0.2);
    R.diagonal().segment(NU0, nxu).array() = tinytype(1.0);
    R.diagonal().segment(NU0 + nxu, nux).array() = tinytype(1.0);
    R.diagonal().segment(NU0 + nxu + nux, nuu).array() = tinytype(10.0);

    Vec fdyn = Vec::Zero(nxL);
    const tinytype rho_base = tinytype(5.0);
    const tinytype rho_psd_penalty = tinytype(0.95);

    Mat x_min = Mat::Constant(nxL, N, -std::numeric_limits<tinytype>::infinity());
    Mat x_max = Mat::Constant(nxL, N,  std::numeric_limits<tinytype>::infinity());
    x_min.topRows(NX0).setConstant(-30.0);
    x_max.topRows(NX0).setConstant( 30.0);
    x_min.middleRows(NX0, NX0 * NX0).setConstant(-1500.0);
    x_max.middleRows(NX0, NX0 * NX0).setConstant( 1500.0);

    Mat u_min = Mat::Constant(nuL, N - 1, -std::numeric_limits<tinytype>::infinity());
    Mat u_max = Mat::Constant(nuL, N - 1,  std::numeric_limits<tinytype>::infinity());
    u_min.topRows(NU0).setConstant(-3.0);
    u_max.topRows(NU0).setConstant( 3.0);
    u_min.bottomRows(nxu + nux + nuu).setConstant(-120.0);
    u_max.bottomRows(nxu + nux + nuu).setConstant( 120.0);

    auto scenarios = build_scenarios();
    std::ofstream csv_summary(output_dir / (cfg.file_prefix + "summary.csv"));
    if (csv_summary.is_open()) {
        csv_summary << "scenario,slug,success,goal_step,min_point_sd,min_seg_sd,min_eta,max_trace_gap,"
                       "all_certified,final_goal_dist,final_vel_norm,planner_solves,planner_total_ms,"
                       "tracker_solves,tracker_total_ms,tracker_avg_us\n";
    }

    int selected_count = 0;
    for (const auto& scenario : scenarios) {
        if (scenario_selected(cfg, scenario.slug)) {
            selected_count++;
        }
    }
    std::cout << "[TinySDP-3D] Running " << selected_count << " dynamic 3D scenario(s)\n";
    int success_count = 0;
    for (const auto& scenario : scenarios) {
        if (!scenario_selected(cfg, scenario.slug)) {
            continue;
        }
        ScenarioResult result = run_scenario(
            scenario, cfg, output_dir, Ad, Bd, A, B, Q, R, x_min, x_max, u_min, u_max, fdyn, rho_base, rho_psd_penalty);
        success_count += result.success ? 1 : 0;
        if (csv_summary.is_open()) {
            csv_summary << result.name << "," << result.slug << ","
                        << (result.success ? 1 : 0) << ","
                        << result.goal_step << ","
                        << result.min_point_sd << ","
                        << result.min_seg_sd << ","
                        << result.min_eta << ","
                        << result.max_trace_gap << ","
                        << (result.all_certified ? 1 : 0) << ","
                        << result.final_goal_dist << ","
                        << result.final_vel_norm << ","
                        << result.planner_solves << ","
                        << result.planner_total_us / 1000.0 << ","
                        << result.tracker_solves << ","
                        << result.tracker_total_us / 1000.0 << ","
                        << (result.tracker_solves > 0 ? result.tracker_total_us / result.tracker_solves : 0.0)
                        << "\n";
        }

        std::cout << "[TinySDP-3D] " << result.slug
                  << " success=" << (result.success ? "YES" : "NO")
                  << " goal_step=" << result.goal_step
                  << " min_seg=" << result.min_seg_sd
                  << " min_eta=" << result.min_eta
                  << " planner_ms=" << result.planner_total_us / 1000.0
                  << " tracker_avg_us=" << (result.tracker_solves > 0
                      ? result.tracker_total_us / result.tracker_solves : 0.0)
                  << "\n";
    }

    std::cout << "[TinySDP-3D] Completed " << success_count << "/" << selected_count
              << " scenarios successfully\n";
    return (success_count == selected_count) ? 0 : 1;
}
