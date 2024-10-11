"""
cafx: afx library with cython
"""
import numpy as np
cimport cython
from libc.math cimport exp, sqrt, abs, pow, M_PI, tan, log10

###############
# CONTROLLERS #
###############
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def envfollower(float[::1] x, 
                float attack_ms=20., 
                float release_ms=200., 
                bint rms=0,
                int sr=44100):
    cdef Py_ssize_t T = len(x)

    cdef float[::1] env = np.zeros(T, dtype=np.float32)
    cdef float exp_factor = -2.0*M_PI*1000/sr
    cdef float att = 0 if attack_ms<1e-3 else exp(exp_factor/attack_ms)
    cdef float rel = exp(exp_factor/release_ms)

    cdef float e = 0.
    cdef float cte = 0.
    cdef float y_t = 0.

    for t in range(T):
        e = x[t]*x[t] if rms else abs(x[t])
        cte = att if e>y_t else rel
        y_t = e+cte*(y_t-e)
        env[t] = sqrt(y_t) if rms else y_t
    return env

#############################
# DYNAMIC RANGE CONTROLLERS #
#############################
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def compressor(float[::1] x,
               float[::1] y, 
               float threshold_db=-18,
               float ratio=4,
               float attack_ms=20,
               float release_ms=200,
               float knee_db=6,
               bint rms=0,
               int sr=44100):

    cdef Py_ssize_t T = len(x)
    cdef float[::1] env = envfollower(y, attack_ms, release_ms, rms, sr)
    cdef float[::1] out = np.zeros(T, dtype=np.float32)

    cdef float gain_db = 0.
    cdef float gain = 0.
    cdef float eps = 1e-7
    cdef float thres_rec = 1/pow(10, threshold_db/20)
    cdef float thres_1 = threshold_db-knee_db/2
    cdef float thres_2 = threshold_db+knee_db/2
    cdef float thres_1_amp = pow(10, thres_1/20)
    cdef float thres_2_amp = pow(10, thres_2/20)
    cdef float r1 = 1/ratio-1
    cdef float R = (r1-1)/2/knee_db

    for t in range(T):
        if env[t] < thres_1_amp:
            out[t] = x[t]
        elif env[t] < thres_2_amp:
            env_db = 20*log10(env[t]+eps)
            gain_db = R*(env_db-thres_1)**2
            gain = pow(10, gain_db*0.05)-eps
            out[t] = x[t]*gain
        else:
            gain = pow(env[t]*thres_rec, r1)
            out[t] = x[t]*gain
    return out

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def noisegate(float[::1] x,
              float[::1] y, 
              float threshold_db=-40,
              float ratio=4,
              float attack_ms=20,
              float release_ms=200,
              float knee_db=6,
              bint rms=0,
              int sr=44100):

    cdef Py_ssize_t T = len(x)
    cdef float[::1] env = envfollower(y, attack_ms, release_ms, rms, sr)
    cdef float[::1] out = np.zeros(T, dtype=np.float32)

    cdef float gain_db = 0.
    cdef float gain = 0.
    cdef float eps = 1e-7
    cdef float thres_rec = 1/pow(10, threshold_db/20)
    cdef float thres_1 = threshold_db-knee_db/2
    cdef float thres_2 = threshold_db+knee_db/2
    cdef float thres_1_amp = pow(10, thres_1/20)
    cdef float thres_2_amp = pow(10, thres_2/20)
    cdef float R = (1-ratio)/2/knee_db
    cdef float r = ratio-1

    # env_db, gain_db, gain, out
    for t in range(T):
        if env[t] < thres_1_amp:
            gain = pow(env[t]*thres_rec, r)
            out[t] = x[t]*gain
        elif env[t] < thres_2_amp:
            env_db = 20*log10(env[t]+eps)
            gain_db = R*(env_db-thres_2)**2
            gain = pow(10, gain_db*0.05)-eps
            out[t] = x[t]*gain
        else:
            out[t] = x[t]
    return out

##################
# LINEAR FILTERS #
##################
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def ltidf2filt(float[::1] x, float G, float twoR, float c_hp, float c_bp, float c_lp):
    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef float b0 = c_hp+c_bp*G+c_lp*G**2
    cdef float b1 = -c_hp*2+c_lp*2*G**2
    cdef float b2 = c_hp-c_bp*G+c_lp*G**2
    cdef float a0 = 1+G**2+twoR*G
    cdef float a1 = 2*G**2-2
    cdef float a2 = 1+G**2-twoR*G
    b0, b1, b2, a1, a2 = b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

    cdef float s1 = 0., s2 = 0.

    for t in range(T):
        y[t] = b0*x[t]+s1
        s1 = b1*x[t]+s2-a1*y[t]
        s2 = b2*x[t]-a2*y[t]
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def tvsvfilt(float[::1] x, float[::1] G, float[::1] twoR, float[::1] c_hp, float[::1] c_bp, float[::1] c_lp):
    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef float s1 = 0., s2 = 0.
    cdef float y_hp = 0., y_bp = 0., y_lp = 0.

    for t in range(T):
        y_bp = (G[t]*(x[t]-s2)+s1)/(1+G[t]*(G[t]+twoR[t]))
        y_lp = G[t]*y_bp+s2
        y_hp = x[t]-y_lp-twoR[t]*y_bp
        y[t] = c_hp[t]*y_hp+c_bp[t]*y_bp+c_lp[t]*y_lp
        s1 = 2*y_bp-s1
        s2 = 2*y_lp-s2
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def lowpass_ladder(float[::1] x, float G=1, float k=0.1):
    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef float S = 0, u = 0
    cdef float G_lp = G/(1+G), G_hp = 1/(1+G), G_lp2 = G_lp**2, G_lp3 = G_lp**3
    cdef float u_div = 1/(1+4*k*G_lp**4)

    cdef float s1 = 0, s2 = 0, s3 = 0, s4 = 0
    cdef float S1 = 0, S2 = 0, S3 = 0, S4 = 0
    cdef float v1 = 0, v2 = 0, v3 = 0, v4 = 0
    cdef float y1 = 0, y2 = 0, y3 = 0

    for t in range(T):
        S1, S2, S3, S4 = s1*G_hp, s2*G_hp, s3*G_hp, s4*G_hp
        S = G_lp3*S1+G_lp2*S2+G_lp*S3+S4
        u = (x[t]-4*k*S)*u_div
        v1 = ( u-s1)*G_lp; y1   = v1+s1; s1 = y1  +v1
        v2 = (y1-s2)*G_lp; y2   = v2+s2; s2 = y2  +v2
        v3 = (y2-s3)*G_lp; y3   = v3+s3; s3 = y3  +v3
        v4 = (y3-s4)*G_lp; y[t] = v4+s4; s4 = y[t]+v4
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def highpass_ladder(float[::1] x, float G=1, float k=0.1):
    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef float k4 = 4*k, G2 = 2*G
    cdef float G_hp = 1/(1+G), G_hp2 = G_hp**2, G_hp3 = G_hp**3
    cdef float u_div = 1/(1+4*k*G_hp**4)

    cdef float S = 0, u = 0
    cdef float s1 = 0, s2 = 0, s3 = 0, s4 = 0
    cdef float S1 = 0, S2 = 0, S3 = 0, S4 = 0
    cdef float v1 = 0, v2 = 0, v3 = 0, v4 = 0
    cdef float y1 = 0, y2 = 0, y3 = 0

    for t in range(T):
        S1, S2, S3, S4 = -s1*G_hp, -s2*G_hp, -s3*G_hp, -s4*G_hp
        S = G_hp3*S1+G_hp2*S2+G_hp*S3+S4
        u = (x[t]-k4*S)*u_div
        v1 =  u-s1; y1   = v1*G_hp; s1 = s1+G2*y1  
        v2 = y1-s2; y2   = v2*G_hp; s2 = s2+G2*y2  
        v3 = y2-s3; y3   = v3*G_hp; s3 = s3+G2*y3  
        v4 = y3-s4; y[t] = v4*G_hp; s4 = s4+G2*y[t]
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def bandpass_ladder(float[::1] x, float G=1, float k=0.1):
    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef float k4 = 4*k, G2 = 2*G
    cdef float G_lp = G/(1+G), G_hp = 1/(1+G)
    cdef float G_hplp = G_hp*G_lp, G_hp2lp = G_hp**2*G_lp, G_hp2lp2 = G_hp**2*G_lp**2
    cdef float u_div = 1/(1-4*k*G_hp2lp2)

    cdef float S = 0, u = 0
    cdef float s1 = 0, s2 = 0, s3 = 0, s4 = 0
    cdef float S1 = 0, S2 = 0, S3 = 0, S4 = 0
    cdef float v1 = 0, v2 = 0, v3 = 0, v4 = 0
    cdef float y1 = 0, y2 = 0, y3 = 0

    for t in range(T):
        S1, S2, S3, S4 = s1*G_hp, -s2*G_hp, s3*G_hp, -s4*G_hp
        S = G_hp2lp*S1+G_hplp*S2+G_hp*S3+S4
        u = (x[t]+k4*S)*u_div
        v1 = ( u-s1)*G_lp; y1   = v1+s1;   s1 = y1+v1
        v2 = y1-s2;        y2   = v2*G_hp; s2 = s2+G2*y2  
        v3 = (y2-s3)*G_lp; y3   = v3+s3;   s3 = y3+v3
        v4 = y3-s4;        y[t] = v4*G_hp; s4 = s4+G2*y[t]
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def tv_lowpass_ladder(float[::1] x, float[::1] G, float k=0.1):
    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef float k4 = 4*k
    cdef float G_lp = 0, G_hp = 0
    cdef float G_lp2 = 0, G_lp3 = 0, G_lp4 = 0

    cdef float S = 0, u = 0
    cdef float s1 = 0, s2 = 0, s3 = 0, s4 = 0
    cdef float S1 = 0, S2 = 0, S3 = 0, S4 = 0
    cdef float v1 = 0, v2 = 0, v3 = 0, v4 = 0
    cdef float y1 = 0, y2 = 0, y3 = 0

    for t in range(T):
        G_hp = 1/(1+G[t]); G_lp = G[t]*G_hp
        G_lp2 = G_lp*G_lp; G_lp3 = G_lp2*G_lp; G_lp4 = G_lp3*G_lp
        S1, S2, S3, S4 = s1*G_hp, s2*G_hp, s3*G_hp, s4*G_hp
        S = G_lp3*S1+G_lp2*S2+G_lp*S3+S4
        u = (x[t]-k4*S)/(1+k4*G_lp4)
        v1 = ( u-s1)*G_lp; y1   = v1+s1; s1 = y1  +v1
        v2 = (y1-s2)*G_lp; y2   = v2+s2; s2 = y2  +v2
        v3 = (y2-s3)*G_lp; y3   = v3+s3; s3 = y3  +v3
        v4 = (y3-s4)*G_lp; y[t] = v4+s4; s4 = y[t]+v4
    return y

######################
# MODULATION EFFECTS #
######################
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def chorus(float[::1] x, 
           float[::1] m, 
           float depth=1,
           float centre_delay_ms=7,
           float feedback=0,
           float mix=0.5,
           float max_delay_ms=25.0,
           int sr=44100):

    cdef Py_ssize_t T = len(x)
    cdef Py_ssize_t C_x = x.shape[-1], C_m = m.shape[-1]

    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef Py_ssize_t L = 1+int(2*sr*centre_delay_ms/1000)
    cdef float[::1] buffer = np.zeros(L, dtype=np.float32)
    cdef Py_ssize_t i = 0
    cdef float ms_to_sample = sr/1000

    cdef float d_ms = 0, d = 0, d_frac = 0
    cdef Py_ssize_t d_int = 0
    cdef float readout = 0

    for t in range(T):
        d_ms = centre_delay_ms*(1+depth*m[t])
        d = d_ms*ms_to_sample
        d_int = int(d); d_frac = d%1
        readout = (1-d_frac)*buffer[(L+i-d_int)%L]+d_frac*buffer[(L+i-d_int-1)%L]
        y[t] = mix*readout+(1-mix)*x[t]
        buffer[i] = x[t]-feedback*readout
        i = (i+1)%L

    return y

##################
# DELAY & REVERB #
##################
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def delay(float[::1] x, 
          float delay_seconds=0.5,
          float feedback_gain_db=-1,
          float mix=0.5,
          float frequency_hz=800,
          float q=2,
          int sr=44100):

    cdef Py_ssize_t T = len(x)
    cdef float[::1] y = np.zeros(T, dtype=np.float32)

    cdef Py_ssize_t L = int(delay_seconds*sr)
    cdef float[::1] buffer = np.zeros(L, dtype=np.float32)
    cdef Py_ssize_t i = 0

    cdef float G = tan(M_PI*frequency_hz/sr), twoR = 1/q
    cdef float b0 = twoR*G, b2 = -twoR*G
    cdef float a0 = 1+G**2+twoR*G, a1 = 2*G**2-2, a2 = 1+G**2-twoR*G
    b0 = b0/a0; b2 = b2/a0; a1 = a1/a0; a2 = a2/a0
    cdef float s1 = 0, s2 = 0, filter_in = 0, filter_out = 0

    cdef float g = 10**(feedback_gain_db/20)

    for t in range(T):
        # filter update
        filter_in = buffer[i]
        filter_out = b0*filter_in+s1
        s1 = s2-a1*filter_out
        s2 = b2*filter_in-a2*filter_out
        # output
        y[t] = mix*filter_out+(1-mix)*x[t]
        # buffer update
        buffer[i] = x[t]+g*filter_out
        i = (i+1)%L
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def pingpong(float[:, ::1] x, 
             float delay_seconds=0.2,
             float lr_ratio=0.5,
             float feedback_gain_db=-1,
             float mix=1,
             float frequency_hz=800,
             float q=2,
             int sr=44100):

    cdef Py_ssize_t T = x.shape[0], C = x.shape[1], C_stereo = 2
    cdef float[:, ::1] y = np.zeros((T, 2), dtype=np.float32)

    cdef Py_ssize_t L = 1+int(delay_seconds*sr)
    cdef Py_ssize_t L_delta = int(delay_seconds*lr_ratio*sr)
    cdef float[::1] buffer = np.zeros(L, dtype=np.float32)
    cdef Py_ssize_t i = 0, left_read = 0

    cdef float G = tan(M_PI*frequency_hz/sr), twoR = 1/q
    cdef float b0 = twoR*G, b2 = -twoR*G
    cdef float a0 = 1+G**2+twoR*G, a1 = 2*G**2-2, a2 = 1+G**2-twoR*G
    b0 = b0/a0; b2 = b2/a0; a1 = a1/a0; a2 = a2/a0
    cdef float s1 = 0, s2 = 0, filter_in = 0, filter_out = 0

    cdef float g = 10**(feedback_gain_db/20)
    cdef float buffer_in = 0

    for t in range(T):
        # output
        left_read = (L+i-L_delta)%L
        y[t, 0] = mix*buffer[left_read]+(1-mix)*x[t, 0]
        y[t, 1] = mix*buffer[i]+(1-mix)*x[t, 1%C]

        # filter update
        filter_in = g*buffer[i]
        for c in range(C):
            filter_in += 0.5*x[t, c]

        filter_out = b0*filter_in+s1
        s1 = s2-a1*filter_out
        s2 = b2*filter_in-a2*filter_out

        # buffer update
        buffer[i] = filter_out
        i = (i+1)%L
    return y

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def freeverb_mono(float[:, ::1] x, float room_size=0.5, float damping=0.5):
    cdef Py_ssize_t T = len(x)
    cdef float feedback = 0.28*room_size+0.7
    damping = damping*0.4
    cdef float[:, ::1] y = np.zeros((T, 1), dtype=np.float32)
    cdef float[:, ::1] buffer_lpcombs = np.zeros((8, 1617), dtype=np.float32)
    cdef Py_ssize_t[::1] i_lpcombs = np.zeros(8, dtype=np.intp)
    cdef Py_ssize_t[::1] L_lpcombs = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], dtype=np.intp)
    cdef float[::1] u_lpcombs = np.zeros(8, dtype=np.float32)
    cdef float lpcomb_sum = 0, lpcomb_out = 0, u = 0, _y = 0, _x = 0
    cdef float[:, ::1] buffer_apfs = np.zeros((4, 556), dtype=np.float32)
    cdef Py_ssize_t[::1] i_apfs = np.zeros(4, dtype=np.intp)
    cdef Py_ssize_t[::1] L_apfs = np.array([556, 441, 341, 225], dtype=np.intp)
    cdef Py_ssize_t C_lpcomb = 8, C_allpass = 4

    for t in range(T):
        lpcomb_sum = 0
        for c in range(C_lpcomb):
            lpcomb_out = feedback*buffer_lpcombs[c, i_lpcombs[c]]
            u_lpcombs[c] = lpcomb_out*(1-damping)+u_lpcombs[c]*damping
            buffer_lpcombs[c, i_lpcombs[c]] = x[t, 0]+feedback*u_lpcombs[c]
            i_lpcombs[c] = (i_lpcombs[c]+1)%L_lpcombs[c]
            lpcomb_sum += lpcomb_out
        _x = lpcomb_sum*0.015
        for c in range(C_allpass):
            u = buffer_apfs[c, i_apfs[c]]
            _y = u-_x
            buffer_apfs[c, i_apfs[c]] = _x+0.5*u
            i_apfs[c] = (i_apfs[c]+1)%L_apfs[c]
            _x = _y
        y[t] = _y
    return y
