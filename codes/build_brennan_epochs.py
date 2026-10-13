"""Brennan FieldTrip .mat -> word-locked epochs, stage1 格式.
每词全局采样 = 段marker(vmrk) + onset*fs. 60 电极, [-0.1,+0.5]s @ 500Hz -> 300 点.
输出 (n_epochs, 60, 300) + trial_metadata (subject/word_text/segment/word_index)."""
import scipy.io as sio, numpy as np, pandas as pd, re, os, glob, json, sys
R="<PROJECT>"
RAW=R+"/data/brennan_raw"; VMRK=R+"/data/brennan_vmrk"
OUT=R+"/data/brennan_p4"; os.makedirs(OUT,exist_ok=True)
fs=500; TMIN,TMAX=-0.1,0.5
w0,w1=int(TMIN*fs),int(TMAX*fs)  # -50, 250 -> 300 点
W=pd.read_csv(R+"/data/brennan_meta/AliceChapterOne-EEG.csv")

def seg_starts(f):
    return [int(m.group(1)) for l in open(f) for m in [re.match(r'Mk\d+=Stimulus,\d+,(\d+),',l)] if m]

def build_subject(sid):
    matf=f"{RAW}/{sid}.mat"; vmrkf=f"{VMRK}/{sid}.vmrk"
    if not (os.path.exists(matf) and os.path.exists(vmrkf)): return None
    try:
        m=sio.loadmat(matf,squeeze_me=True,struct_as_record=False)
        X=m["raw"].trial  # (62, T)
    except Exception as e:
        print(f"  {sid} loadmat 失败: {str(e)[:40]}"); return None
    elec=X[:60].astype(np.float32); N=X.shape[1]
    starts=seg_starts(vmrkf)
    if len(starts)<12: print(f"  {sid} marker 不足: {len(starts)}"); return None
    eps=[]; meta=[]
    for s in range(1,13):
        ss=W[W.Segment==s]
        for _,r in ss.iterrows():
            t=int(starts[s-1]+r.onset*fs)
            if t+w0<0 or t+w1>N: continue
            eps.append(elec[:, t+w0:t+w1])
            meta.append((sid, str(r.Word), int(s), int(r.Order)))
    if not eps: return None
    return np.stack(eps).astype(np.float32), meta

if __name__=="__main__":
    subs=sorted(set(os.path.basename(f)[:3] for f in glob.glob(RAW+"/S*.mat")))
    all_ep=[]; all_meta=[]
    for sid in subs:
        res=build_subject(sid)
        if res is None: continue
        ep,meta=res; all_ep.append(ep); all_meta+=meta
        print(f"  {sid}: {ep.shape[0]} epochs", flush=True)
    if all_ep:
        EP=np.concatenate(all_ep,0)
        md=pd.DataFrame(all_meta,columns=["subject","word_text","segment","word_index"])
        np.save(OUT+"/epochs_data.npy",EP)
        md.to_csv(OUT+"/trial_metadata.tsv",sep="\t",index=False)
        print(f"\n总: {EP.shape} epochs, {md.subject.nunique()} 被试, {len(md)} 词-试次")
        print(f"保存到 {OUT}")
