import numpy as np
import pandas as pd
EFFECT_ORDER = [
    "noise",
    "bandlimit",
    "EQ",
    "delay",
    "mono_reverb",
    "ir",
    "compressors",
    "clipping",
    "distortion",
    "modulation",
    "codec",
    "monolithic",
    "complex",
]
MERGE = {
    "ir": ["micir_conv", "rir_conv"],
    "bandlimit": ["lowpass", "bandpass", "highpass", "bandreject"],
    "compressors": ["compressor", "limiter"],
}
def merge(df):
    for k, v in MERGE.items():
        list = [df.loc[k_] for k_ in v]
        df.loc[k] = sum(list) / len(list)
        df.drop(v, inplace=True)
    return df
def exp_1():
    train_single_eval_single = pd.read_csv(
        "train_single-eval_single_vctk1__metric_summary_pred.csv"
    )
    train_single_eval_multi = pd.read_csv(
        "train_single-eval_multi_vctk1__metric_summary_pred.csv"
    )
    train_single = pd.concat([train_single_eval_single, train_single_eval_multi])
    train_single = train_single.set_index("Effect Type")
    train_single = merge(train_single)
    train_single = train_single.reindex(EFFECT_ORDER)
    train_multi_eval_single = pd.read_csv(
        "train_multi-eval_single_vctk1__metric_summary_pred.csv"
    )
    train_multi_eval_multi = pd.read_csv(
        "train_multi-eval_multi_vctk1__metric_summary_pred.csv"
    )
    train_multi = pd.concat([train_multi_eval_single, train_multi_eval_multi])
    train_multi = train_multi.set_index("Effect Type")
    train_multi = merge(train_multi)
    train_multi = train_multi.reindex(EFFECT_ORDER)
    dry_eval_multi = pd.read_csv("train_multi-eval_multi_vctk1__metric_summary_dry.csv")
    dry_eval_single = pd.read_csv(
        "train_single-eval_single_vctk1__metric_summary_dry.csv"
    )
    dry = pd.concat([dry_eval_single, dry_eval_multi])
    dry = dry.set_index("Effect Type")
    dry = merge(dry)
    dry = dry.reindex(EFFECT_ORDER)
    print(train_single)
    for effect in EFFECT_ORDER:
        strs = []
        print(effect)
        for metric in ["si_sdr", "sc_loss", "lsm_loss"]:
            dry_result = dry.loc[effect, metric]
            single_result = train_single.loc[effect, metric]
            multi_result = train_multi.loc[effect, metric]
            if metric in ["sc_loss", "lsm_loss"]:
                best = np.min([dry_result, single_result, multi_result])
            else:
                best = np.max([dry_result, single_result, multi_result])
            if best == dry_result:
                strs.append("$\mathbf{%0.2f}$" % dry_result)
            else:
                strs.append("$%0.2f$" % dry_result)
            if best == single_result:
                strs.append("$\mathbf{%0.2f}$" % single_result)
            else:
                strs.append("$%0.2f$" % single_result)
            if best == multi_result:
                strs.append("$\mathbf{%0.2f}$" % multi_result)
            else:
                strs.append("$%0.2f$" % multi_result)
        str = " & ".join(strs)
        print(str)
        print("")

def exp_2():
    single_daps = pd.read_csv("single_recording_env_daps__metric_summary_pred.csv")
    single_vctk1 = pd.read_csv("single_recording_env_vctk1__metric_summary_pred.csv")
    single_vctk2 = pd.read_csv("single_recording_env_vctk2__metric_summary_pred.csv")
    single_daps = (
        merge(single_daps.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    single_vctk1 = (
        merge(single_vctk1.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    single_vctk2 = (
        merge(single_vctk2.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    multi_daps = pd.read_csv("multiple_recording_env_daps__metric_summary_pred.csv")
    multi_vctk1 = pd.read_csv("multiple_recording_env_vctk1__metric_summary_pred.csv")
    multi_vctk2 = pd.read_csv("multiple_recording_env_vctk2__metric_summary_pred.csv")
    multi_daps = (
        merge(multi_daps.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    multi_vctk1 = (
        merge(multi_vctk1.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    multi_vctk2 = (
        merge(multi_vctk2.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    dry_daps = pd.read_csv("single_recording_env_daps__metric_summary_dry.csv")
    dry_vctk1 = pd.read_csv("single_recording_env_vctk1__metric_summary_dry.csv")
    dry_vctk2 = pd.read_csv("single_recording_env_vctk2__metric_summary_dry.csv")
    dry_daps = (
        merge(dry_daps.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    dry_vctk1 = (
        merge(dry_vctk1.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    dry_vctk2 = (
        merge(dry_vctk2.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2]).mean()
    )
    strs = []
    for metric in ["si_sdr", "sc_loss", "lsm_loss"]:
        dry_result = dry_daps[metric]
        single_result = single_daps[metric]
        multi_result = multi_daps[metric]
        if metric in ["sc_loss", "lsm_loss"]:
            best = np.min([dry_result, single_result, multi_result])
        else:
            best = np.max([dry_result, single_result, multi_result])
        if best == dry_result:
            strs.append("$\mathbf{%0.2f}$" % dry_result)
        else:
            strs.append("$%0.2f$" % dry_result)
        if best == single_result:
            strs.append("$\mathbf{%0.2f}$" % single_result)
        else:
            strs.append("$%0.2f$" % single_result)
        if best == multi_result:
            strs.append("$\mathbf{%0.2f}$" % multi_result)
        else:
            strs.append("$%0.2f$" % multi_result)
    str = " & ".join(strs)
    print(str)
    print("")
    strs = []
    for metric in ["si_sdr", "sc_loss", "lsm_loss"]:
        dry_result = dry_vctk1[metric]
        single_result = single_vctk1[metric]
        multi_result = multi_vctk1[metric]
        if metric in ["sc_loss", "lsm_loss"]:
            best = np.min([dry_result, single_result, multi_result])
        else:
            best = np.max([dry_result, single_result, multi_result])
        if best == dry_result:
            strs.append("$\mathbf{%0.2f}$" % dry_result)
        else:
            strs.append("$%0.2f$" % dry_result)
        if best == single_result:
            strs.append("$\mathbf{%0.2f}$" % single_result)
        else:
            strs.append("$%0.2f$" % single_result)
        if best == multi_result:
            strs.append("$\mathbf{%0.2f}$" % multi_result)
        else:
            strs.append("$%0.2f$" % multi_result)
    str = " & ".join(strs)
    print(str)
    print("")
    strs = []
    for metric in ["si_sdr", "sc_loss", "lsm_loss"]:
        dry_result = dry_vctk2[metric]
        single_result = single_vctk2[metric]
        multi_result = multi_vctk2[metric]
        if metric in ["sc_loss", "lsm_loss"]:
            best = np.min([dry_result, single_result, multi_result])
        else:
            best = np.max([dry_result, single_result, multi_result])
        if best == dry_result:
            strs.append("$\mathbf{%0.2f}$" % dry_result)
        else:
            strs.append("$%0.2f$" % dry_result)
        if best == single_result:
            strs.append("$\mathbf{%0.2f}$" % single_result)
        else:
            strs.append("$%0.2f$" % single_result)
        if best == multi_result:
            strs.append("$\mathbf{%0.2f}$" % multi_result)
        else:
            strs.append("$%0.2f$" % multi_result)
    str = " & ".join(strs)
    print(str)
    print("")
def exp_3():
    dry = pd.read_csv("cross_domain_maestro_single__metric_summary_dry.csv")
    pred = pd.read_csv("cross_domain_maestro_single__metric_summary_pred.csv")
    dry = merge(dry.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2])
    pred = merge(pred.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2])
    single_daps = pd.read_csv("single_recording_env_daps__metric_summary_pred.csv")
    single_vctk1 = pd.read_csv("single_recording_env_vctk1__metric_summary_pred.csv")
    single_vctk2 = pd.read_csv("single_recording_env_vctk2__metric_summary_pred.csv")
    single_daps = merge(single_daps.set_index("Effect Type")).reindex(EFFECT_ORDER[:-2])
    single_vctk1 = merge(single_vctk1.set_index("Effect Type")).reindex(
        EFFECT_ORDER[:-2]
    )
    single_vctk2 = merge(single_vctk2.set_index("Effect Type")).reindex(
        EFFECT_ORDER[:-2]
    )
    for effect in EFFECT_ORDER[:-2]:
        strs = []
        print(effect)
        for metric in ["si_sdr", "sc_loss", "lsm_loss"]:
            dry_result = dry.loc[effect, metric]
            single_result = np.mean(
                [
                    single_daps.loc[effect, metric],
                    single_vctk1.loc[effect, metric],
                    single_vctk2.loc[effect, metric],
                ]
            )
            result = pred.loc[effect, metric]
            if metric in ["sc_loss", "lsm_loss"]:
                best = np.min([dry_result, result, single_result])
            else:
                best = np.max([dry_result, result, single_result])
            if best == dry_result:
                strs.append("$\mathbf{%0.2f}$" % dry_result)
            else:
                strs.append("$%0.2f$" % dry_result)
            if best == single_result:
                strs.append("$\mathbf{%0.2f}$" % single_result)
            else:
                strs.append("$%0.2f$" % single_result)
            if best == result:
                strs.append("$\mathbf{%0.2f}$" % result)
            else:
                strs.append("$%0.2f$" % result)
        str = " & ".join(strs)
        print(str)
        print("")
if __name__ == "__main__":
#     exp_1()
    exp_2()
#     exp_3()
