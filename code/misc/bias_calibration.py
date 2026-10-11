import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.size':10,'mathtext.fontset':'cm',
                     'axes.linewidth':0.7,'lines.linewidth':1.5})
BLUE='#2f4b9c'; ORANGE='#d1731c'; RED='#b02a2a'; GRAY='#555'

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../out/figures/bias_calibration.pdf')

widths=[1,64,64,64,64]                 # 1D input + four hidden layers (used for panels A-D)
xg=np.linspace(-1,1,400)
xf=np.linspace(-1,1,600)               # finer grid for the prior draws
layers=np.arange(1,len(widths))
SEEDS=400

# hierarchical GSBL draw of a variance: psi ~ GG(r,beta,vartheta), sampled as vartheta * G^{1/r}
def sample_psi(shape,r,beta,vth,rng):
    return vth*rng.gamma(beta,1.0,size=shape)**(1.0/r)

def fanin_exp(r,beta):
    alpha=2*abs(r)*beta
    return 1.0 if alpha>=2 else 2.0/alpha            # 1/n (finite var) or n^{-2/alpha}

# panels A-D: per-layer median pre-activation spread, active fraction, transition locations
def run(r,beta):
    expo=fanin_exp(r,beta)
    Spool=[[] for _ in layers]; frac=[[] for _ in layers]; cross=[[] for _ in layers]
    for seed in range(SEEDS):
        rng=np.random.default_rng(seed); h=xg[None,:].copy()
        for li,l in enumerate(range(1,len(widths))):
            fi=widths[l-1]; no=widths[l]; th=fi**(-expo)
            psi=sample_psi((no,fi),r,beta,th,rng)
            W=rng.normal(0,1,size=(no,fi))*np.sqrt(psi)
            b=rng.normal(0,np.sqrt(th),size=(no,1))
            zpre=W@h
            Spool[li].extend(np.std(zpre,axis=1).tolist())
            z=zpre+b
            s=np.sign(z); mask=s[:,:-1]*s[:,1:]<0
            frac[li].append(np.mean(mask.any(axis=1)))
            ii,kk=np.where(mask)
            xc=xg[kk]-z[ii,kk]*(xg[kk+1]-xg[kk])/(z[ii,kk+1]-z[ii,kk])
            cross[li].append(xc[(xc>=-1)&(xc<=1)])
            h=np.tanh(z)
    return ([np.median(s) for s in Spool],
            [100*np.mean(f) for f in frac],
            [np.concatenate(c) for c in cross])

# panels E-F: prior draw of the scalar network output u_theta(x)
def draw_u(r,beta,seed,arch=(1,64,64,64,64,1)):
    expo=fanin_exp(r,beta); rng=np.random.default_rng(seed); h=xf[None,:].copy(); Ln=len(arch)-1
    for l in range(1,len(arch)):
        fi=arch[l-1]; no=arch[l]; th=fi**(-expo)
        psi=sample_psi((no,fi),r,beta,th,rng)
        W=rng.normal(0,1,size=(no,fi))*np.sqrt(psi)
        b=rng.normal(0,np.sqrt(th),size=(no,1))
        z=W@h+b; h=np.tanh(z) if l<Ln else z
    return h[0]

priors=[(r'Laplace ($r{=}1,\,\beta{=}1$)', 1, 1.0, BLUE,  'o','-'),
        (r'Student-$t$ ($\alpha{=}3$)',    -1, 1.5, ORANGE,'s','-'),
        (r'Cauchy ($\alpha{=}1$)',         -1, 0.5, RED,   '^','--')]
D={lab:run(r,b) for (lab,r,b,c,m,ls) in priors}
LSEEDS=[0,1,2,3,4,5]                       # representative order-1 Laplace draws
CSEEDS=[32,44,68,127,195,123]             # representative order-1 Cauchy draws with sharp jumps

fig,axes=plt.subplots(3,2,figsize=(8.2,8.7))
fig.subplots_adjust(left=0.09,right=0.985,bottom=0.06,top=0.92,wspace=0.28,hspace=0.42)

# (A) median pre-activation spread
ax=axes[0,0]
for lab,r,b,c,m,ls in priors:
    ax.plot(layers,D[lab][0],marker=m,ls=ls,color=c,label=lab)
ax.set_xticks(layers); ax.set_xlabel(r'layer $\ell$')
ax.set_ylabel(r'median pre-activation spread $S_\ell$')
ax.set_title(r'(A) contraction is a finite-variance effect',fontsize=9.5); ax.set_ylim(0,0.95)

# (B) fraction transitioning within Omega
ax=axes[0,1]
for lab,r,b,c,m,ls in priors:
    ax.plot(layers,D[lab][1],marker=m,ls=ls,color=c,label=lab)
ax.set_xticks(layers); ax.set_xlabel(r'layer $\ell$')
ax.set_ylabel(r'neurons transitioning in $\Omega$ [%]')
ax.set_title(r'(B) per-layer bias stays matched at every depth',fontsize=9.5); ax.set_ylim(0,100)

# (C)/(D) transition-location distributions
def violin_panel(ax,lab,color,tag):
    cpool=D[lab][2]; fmean=D[lab][1]
    ax.axhspan(-1,1,color=GRAY,alpha=0.10,lw=0)
    parts=ax.violinplot([c for c in cpool],positions=layers,widths=0.72,showextrema=False)
    for pc in parts['bodies']:
        v=pc.get_paths()[0].vertices; v[:,1]=np.clip(v[:,1],-1,1)
        pc.set_facecolor(color); pc.set_edgecolor(color); pc.set_alpha(0.45)
    ax.plot(layers,[np.median(c) for c in cpool],'_',color=color,ms=12,mew=2)
    for l,f in zip(layers,fmean):
        ax.text(l,1.1,f'{f:.0f}%',ha='center',va='bottom',fontsize=7.6,color=color)
    ax.set_xticks(layers); ax.set_xlabel(r'layer $\ell$')
    ax.set_ylabel(r'transition location in $\Omega$'); ax.set_ylim(-1.28,1.42)
    ax.set_yticks([-1,0,1]); ax.set_yticklabels([r'$-1$','$0$','$1$']); ax.set_title(tag,fontsize=9.5)
violin_panel(axes[1,0],r'Laplace ($r{=}1,\,\beta{=}1$)',BLUE,r'(C) transition locations, Laplace')
violin_panel(axes[1,1],r'Cauchy ($\alpha{=}1$)',RED,r'(D) transition locations, Cauchy')

# (E)/(F) prior draws of u_theta
def draw_panel(ax,r,beta,seeds,color,tag):
    for s in seeds:
        ax.plot(xf,draw_u(r,beta,s),color=color,lw=1.1,alpha=0.85)
    ax.axhline(0,color='k',lw=0.4,alpha=0.3)
    ax.set_xlim(-1,1); ax.set_ylim(-1.35,1.35); ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u_{\boldsymbol{\theta}}(x)$'); ax.set_title(tag,fontsize=9.5)
draw_panel(axes[2,0],1,1.0,LSEEDS,BLUE,r'(E) prior draws $u_{\boldsymbol{\theta}}$, Laplace')
draw_panel(axes[2,1],-1,0.5,CSEEDS,RED,r'(F) prior draws $u_{\boldsymbol{\theta}}$, Cauchy')

handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8.4,frameon=False,
           bbox_to_anchor=(0.5,0.975),handlelength=2.2,columnspacing=1.6)

fig.savefig(OUT,bbox_inches='tight',pad_inches=0.02)
print("saved",OUT)
