// bb_core.c
// Minimal C port of your step() integrator + reward for an inverting buck-boost.
// Exposes: create/destroy/reset/step. Doubles throughout for numerical stability.

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef BB_EXPORT
#  ifdef _WIN32
#    define BB_EXPORT __declspec(dllexport)
#  else
#    define BB_EXPORT __attribute__((visibility("default")))
#  endif
#endif

typedef struct {
    // --- params ---
    double dt;       // substep
    int    N;        // frame_skip
    int    max_episode_steps;
    int    grace_period_steps;

    double target_voltage;

    double Vin, L, C, R;
    double Ron_sw, R_L;
    double Vf, Ron_d;
    double R_C;
    double G_off;

    int    enable_rr;
    double Qrr;
    double trr;

    double duty_min, duty_max;

    // clamps / limits
    double Tsw;
    double I_L_MAX;
    double V_OUT_MAX;
    double V_OUT_MIN;
    double V_CLAMP;
    double I_CLAMP;

    // --- state ---
    double time;
    int    step_count;

    double iL;
    double uC;
    double vC;

    double prev_vC;
    double prev_error;
    double prev_applied_duty;

    // reverse recovery
    double rr_timer;
    double rr_i0;
    double rr_elapsed;
} BBEnv;

static inline double clamp(double x, double lo, double hi){
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline double alpha(const BBEnv* e){
    // alpha = 1 + R_C / R
    double denom = (fabs(e->R) < 1e-12) ? (e->R >= 0 ? 1e-12 : -1e-12) : e->R;
    return 1.0 + (e->R_C / denom);
}

static void on_step(BBEnv* e, double dt, double* iL, double* uC, double* v)
{
    // ON: diode off, inductor to Vin via Ron_sw + R_L; capacitor discharges into load
    double a = alpha(e);
    double v_node = (*uC) / a;

    double diL = (e->Vin - (*iL)*(e->Ron_sw + e->R_L)) / e->L;
    double i_leak = e->G_off * v_node;
    double iC = - v_node / e->R - i_leak;
    double duC = iC / e->C;

    *iL += dt * diL;
    *uC += dt * duC;
    *v   = (*uC) / a;
}

static void off_step(BBEnv* e, double dt, double* iL, double* uC, double* v)
{
    // OFF: inductor discharges through diode into node
    double irr = 0.0;
    if (e->enable_rr && e->rr_timer > 0.0){
        double trr = (e->trr <= 1e-12) ? 1e-12 : e->trr;
        irr = - e->rr_i0 * exp(-(e->rr_elapsed / trr));
        e->rr_elapsed += dt;
        if (e->rr_elapsed >= e->trr){
            e->rr_timer = 0.0;
            irr = 0.0;
        }
    }

    double a = alpha(e);
    double v_node = ((*uC) - e->R_C * ((*iL) + irr)) / a;

    double Rseries = e->Ron_d + e->R_L;
    double diL = (v_node - e->Vf - (*iL)*Rseries) / e->L;
    double iL_pred = (*iL) + dt*diL;

    // DCM split if current crosses zero this substep
    if ((diL < 0.0) && (*iL > 0.0) && (iL_pred < 0.0)){
        double dt1 = (*iL)/(-diL);
        double i_leak = e->G_off * v_node;
        double iC1 = -((*iL) + irr) - v_node/e->R - i_leak;
        double duC1 = iC1 / e->C;
        double uC1 = (*uC) + dt1*duC1;

        // start reverse recovery pulse when turning off
        if (e->enable_rr && e->Qrr > 0 && e->trr > 0){
            e->rr_timer = 1.0;
            e->rr_elapsed = 0.0;
            e->rr_i0 = (e->Qrr / e->trr) * 2.0;
        }

        double dt2 = dt - dt1;
        double uC2 = uC1;
        if (dt2 > 0.0){
            double v_mid = uC1 / a;
            double i_leak2 = e->G_off * v_mid;
            double iC2 = -(0.0 + irr) - v_mid/e->R - i_leak2;
            double duC2 = iC2 / e->C;
            uC2 = uC1 + dt2*duC2;
        }

        *iL = 0.0;
        *uC = uC2;
        *v  = (*uC) / a;
        return;
    }

    // no zero-cross
    double i_leak = e->G_off * v_node;
    double iC = -((*iL) + irr) - v_node/e->R - i_leak;
    double duC = iC / e->C;

    *uC += dt * duC;
    *iL  = iL_pred;

    if (*iL < 0.0){ // guard
        *iL = 0.0;
        *v  = (*uC) / a;
    } else {
        *v  = ((*uC) - e->R_C * ((*iL) + irr)) / a;
    }
}

// reward identical to your Python version
static double reward_fn(BBEnv* e, double duty_eff)
{
    double v_abs = fabs(e->vC);
    double vref  = fabs(e->target_voltage);
    double denom = (vref < 1e-9) ? 1e-9 : vref;

    double eabs = fabs(v_abs - vref);
    double e_norm = eabs / denom;

    double exparg = -5.0 * (e_norm * e_norm);
    double r_track = (exparg < -50.0) ? 0.0 : exp(exparg);

    double prev_v_abs = fabs(e->prev_vC);
    double prev_norm  = prev_v_abs / denom;
    double progress   = (prev_norm - e_norm);

    double band_bonus = (fabs(v_abs - vref) <= 0.02 * vref) ? 0.1 : 0.0;

    double dduty = (isnan(e->prev_applied_duty)) ? 0.0 : (duty_eff - e->prev_applied_duty);

    double r = r_track + 0.1*progress + band_bonus - 0.5*(dduty*dduty);
    if (r < -5.0) r = -5.0;
    if (r >  2.0) r =  2.0;
    return r;
}

BB_EXPORT BBEnv* bb_create(
    double dt, int N, int max_episode_steps, int grace_period_steps,
    double target_voltage,
    double Vin, double L, double C, double R_load,
    double Ron_sw, double R_L, double Vf, double Ron_d, double R_C,
    double G_off, int enable_rr, double Qrr, double trr,
    double duty_min, double duty_max
){
    BBEnv* e = (BBEnv*)calloc(1, sizeof(BBEnv));
    e->dt = dt; e->N = N; e->max_episode_steps = max_episode_steps; e->grace_period_steps = grace_period_steps;
    e->target_voltage = target_voltage;
    e->Vin = Vin; e->L = L; e->C = C; e->R = R_load;
    e->Ron_sw = Ron_sw; e->R_L = R_L; e->Vf = Vf; e->Ron_d = Ron_d; e->R_C = R_C;
    e->G_off = G_off; e->enable_rr = enable_rr; e->Qrr = Qrr; e->trr = trr;
    e->duty_min = duty_min; e->duty_max = duty_max;

    e->Tsw = e->dt * (double)e->N;
    e->I_L_MAX = 20.0;
    e->V_OUT_MAX = fabs(e->target_voltage) * 2.0;
    e->V_OUT_MIN = fabs(e->target_voltage) * 0.05;
    e->V_CLAMP = 1e6;
    e->I_CLAMP = 1e3;

    e->prev_applied_duty = NAN; // match Python “None”
    return e;
}

BB_EXPORT void bb_destroy(BBEnv* e){
    if (e) free(e);
}

BB_EXPORT void bb_reset(BBEnv* e, double* obs4_out) {
    e->time = 0.0;
    e->step_count = 0;
    e->rr_timer = 0.0;
    e->rr_elapsed = 0.0;

    e->iL = 0.0;
    e->uC = 0.0;
    e->vC = 0.0;

    e->prev_vC = e->vC;
    e->prev_error = e->vC - e->target_voltage;
    e->prev_applied_duty = 0.0;

    // obs = [vC, error, d_error, target]
    obs4_out[0] = e->vC;
    obs4_out[1] = e->prev_error;
    obs4_out[2] = 0.0;
    obs4_out[3] = e->target_voltage;
}

// returns obs[4], reward, terminated, truncated
BB_EXPORT void bb_step(BBEnv* e, double duty_cmd,
                       double* obs4_out, double* reward_out,
                       int* term_out, int* trunc_out)
{
    double duty = clamp(duty_cmd, e->duty_min, e->duty_max);
    double x = duty * (double)e->N;
    int    k_on = (int)floor(x + 1e-12);
    double f    = x - (double)k_on;
    double duty_eff = duty; // unquantized

    double iL = e->iL, uC = e->uC, v = e->vC;

    e->rr_timer = 0.0;

    // period means (not required; could be logged if desired)
    // Integrate ON slots
    for (int k = 0; k < k_on; ++k){
        on_step(e, e->dt, &iL, &uC, &v);
        e->time += e->dt;
    }

    if (f > 1e-12){
        on_step(e, e->dt * f, &iL, &uC, &v);
        e->time += e->dt * f;
        off_step(e, e->dt * (1.0 - f), &iL, &uC, &v);
        e->time += e->dt * (1.0 - f);
        for (int k = 0; k < (e->N - k_on - 1); ++k){
            off_step(e, e->dt, &iL, &uC, &v);
            e->time += e->dt;
        }
    } else {
        for (int k = 0; k < (e->N - k_on); ++k){
            off_step(e, e->dt, &iL, &uC, &v);
            e->time += e->dt;
        }
    }

    // clamp and commit
    e->iL = clamp(iL, -e->I_CLAMP, e->I_CLAMP);
    e->uC = clamp(uC, -e->V_CLAMP, e->V_CLAMP);
    e->vC = clamp(v , -e->V_CLAMP, e->V_CLAMP);

    e->step_count += 1;

    double error = e->vC - e->target_voltage;
    double d_error = (e->Tsw > 0) ? (error - e->prev_error)/e->Tsw : 0.0;

    // reward
    double r = reward_fn(e, duty_eff);

    // termination / truncation
    int terminated = 0, truncated = 0;
    if (e->step_count > e->grace_period_steps){
        if (fabs(e->iL) > e->I_L_MAX ||
            !(e->V_OUT_MIN <= fabs(e->vC) && fabs(e->vC) <= e->V_OUT_MAX))
        {
            r -= 1000.0;
            terminated = 1;
        }
    }
    if (!terminated && e->step_count >= e->max_episode_steps){
        truncated = 1;
    }

    // obs
    obs4_out[0] = e->vC;
    obs4_out[1] = error;
    obs4_out[2] = d_error;
    obs4_out[3] = e->target_voltage;

    *reward_out = r;
    *term_out   = terminated;
    *trunc_out  = truncated;

    e->prev_error = error;
    e->prev_applied_duty = duty_eff;
    e->prev_vC = e->vC;
}
