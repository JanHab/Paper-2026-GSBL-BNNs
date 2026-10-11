import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import gamma as G

plt.rcParams.update({'font.family':'serif','font.size':11,'mathtext.fontset':'cm',
                     'axes.linewidth':0.6,'lines.linewidth':1.2})
rng=np.random.default_rng(5)
BLUE='#2f4b9c'; BLUEM='#3f74bf'; BLUEL='#7aa2d8'; RED='#b02a2a'; GREEN='#1a7a3a'; BLACK='#111'; GRAY='#555'
FILLB='#f3f7fd'; FILLR='#fdf4f4'; FILLG='#f2f9f4'
r,beta,vth=-1.0,0.7,0.05
TS=12.5; PT=11; AL=10.5
PANEL_H=0.175

fig=plt.figure(figsize=(7.8,4.35))
def group(x0,y0,x1,y1,edge,fill,lw=1.7):
    fig.patches.append(FancyBboxPatch((x0,y0),x1-x0,y1-y0,
        boxstyle='round,pad=0.004,rounding_size=0.02',transform=fig.transFigure,
        facecolor=fill,edgecolor=edge,lw=lw,linestyle=(0,(6,4)),zorder=-1))
def arrow(xc,y,L=0.030,color=BLUE):
    fig.patches.append(FancyArrowPatch((xc-L/2,y),(xc+L/2,y),transform=fig.transFigure,
        arrowstyle='-|>',mutation_scale=13,lw=1.3,color=color))
def inset(l,b,w,h):
    ax=fig.add_axes([l,b,w,h]); ax.set_xticks([]); ax.set_yticks([]); ax.patch.set_alpha(0)
    for s in ax.spines.values(): s.set_linewidth(0.6); s.set_color(GRAY)
    return ax
def layout(T):
    ptop=T-0.1435
    return dict(title=T-0.042, ptop=ptop, pbot=ptop-PANEL_H,
                phead=ptop+0.007, xlab=ptop-PANEL_H-0.006, bottom=ptop-PANEL_H-0.054)

def sample_u(xs,H=24,seed=None,ascale=0.35,omscale=4.0,adf=2):
    rr=np.random.default_rng(seed)
    xi=rr.uniform(-1,1,H); om=rr.standard_t(1,H)*omscale; a=rr.standard_t(adf,H)*ascale
    return np.sum(a[:,None]*np.tanh(om[:,None]*(xs[None,:]-xi[:,None])),axis=0)
def utrue(x,j1=-0.4,j2=0.35,wig=1.0):
    x=np.asarray(x,float); u=np.empty_like(x)
    m1=x<j1; m2=(x>=j1)&(x<j2); m3=x>=j2
    u[m1]=0.25+0.18*np.sin(2.5*(x[m1]+1))*wig
    u[m2]=0.95-0.22*(x[m2]-j1)+0.05*np.cos(4*x[m2])*wig
    u[m3]=0.45+0.30*(x[m3]-j2)**2*wig
    return u
xs=np.linspace(-1,1,700)
def plot_pw(ax,j1,j2,color,lw=1.4,z=2,wig=1.0,al=1.0):
    for lo,hi in [(-1,j1),(j1,j2),(j2,1)]:
        xseg=np.linspace(lo,hi,250)
        ax.plot(xseg,utrue(xseg,j1,j2,wig),color=color,lw=lw,zorder=z,alpha=al,solid_capstyle='round')

# ===== TOP: prior box =====
Tp=0.975; L=layout(Tp)
group(0.045,L['bottom'],0.965,Tp,BLUE,FILLB)
fig.text(0.505,L['title'],r'edge-preserving GSBL-BNN prior $\pi^0(\boldsymbol{\theta},\boldsymbol{\psi})$',ha='center',va='center',fontsize=TS,color=BLUE)
ww=0.180; xl=[0.070,0.300,0.530,0.760]
titles=[r'$\mathcal{GG}(r,\beta,\vartheta_i)$',r'$\mathcal{N}(0,\psi_i)$',r'$\boldsymbol{\theta}$',r'$u_{\boldsymbol{\theta}}$']
xlabs=[r'$\psi$',r'$\theta$',r'$i$',r'$x$']
ax=inset(xl[0],L['pbot'],ww,PANEL_H); th=np.linspace(1e-3,0.5,400)
dens=np.abs(r)/(G(beta)*vth)*(th/vth)**(r*beta-1)*np.exp(-(th/vth)**r)
ax.plot(th,dens,color=BLUE); ax.fill_between(th,dens,color=BLUE,alpha=0.14); ax.axvline(0.09,color=GRAY,ls=':',lw=0.9); ax.set_xlim(0,0.5)
ax=inset(xl[1],L['pbot'],ww,PANEL_H); w=np.linspace(-1.2,1.2,300)
for tt,c in [(0.09,BLUE),(0.4,BLUEL)]: ax.plot(w,np.exp(-w**2/(2*tt))/np.sqrt(2*np.pi*tt),color=c)
ax=inset(xl[2],L['pbot'],ww,PANEL_H); rs=np.random.default_rng(16); K=20
theta_s=vth*rs.standard_gamma(beta,K)**(1.0/r); w_s=rs.normal(0,np.sqrt(theta_s))
ml,sl,bl=ax.stem(np.arange(1,K+1),w_s,linefmt='-',markerfmt='o',basefmt=' ')
plt.setp(sl,color=BLUEM,lw=1.0); plt.setp(ml,color=BLUE,ms=2.8); ax.axhline(0,color=GRAY,lw=0.5)
ax=inset(xl[3],L['pbot'],ww,PANEL_H)
for s,c in [(287,BLUE),(125,BLUEM),(246,BLUEL)]: ax.plot(xs,sample_u(xs,seed=s),color=c,lw=1.0)
for x,t,xlb in zip(xl,titles,xlabs):
    fig.text(x+ww/2,L['phead'],t,ha='center',va='bottom',fontsize=PT)
    fig.text(x+ww/2,L['xlab'],xlb,ha='center',va='top',fontsize=AL)
yc=(L['ptop']+L['pbot'])/2
for i in range(3): arrow((xl[i]+ww+xl[i+1])/2,yc)

# ===== BOTTOM: + data = posterior (same height, consistent spacing) =====
Tb=L['bottom']-0.070; Lb=layout(Tb)
ycb=(Lb['ptop']+Lb['pbot'])/2
fig.text(0.030,ycb,r'$\cdot$',ha='center',va='center',fontsize=46,color=BLACK)
fig.text(0.508,ycb,r'$\propto$',ha='center',va='center',fontsize=20,color=BLACK)
group(0.075,Lb['bottom'],0.475,Tb,RED,FILLR)
group(0.545,Lb['bottom'],0.965,Tb,GREEN,FILLG)
fig.text(0.275,Lb['title'],r'likelihood $f(\boldsymbol{\theta};\mathbf{y})$',ha='center',va='center',fontsize=TS,color=RED)
fig.text(0.755,Lb['title'],r'GSBL-BNN posterior $\pi^{\mathbf{y}}(\boldsymbol{\theta},\boldsymbol{\psi})$',ha='center',va='center',fontsize=TS,color=GREEN)
fig.text(0.275,Lb['phead'],r'data $\mathbf{y}=F(u)+\mathbf{e}$',ha='center',va='bottom',fontsize=PT)
fig.text(0.755,Lb['phead'],r'$u_{\boldsymbol{\theta}}(x)$ samples',ha='center',va='bottom',fontsize=PT)
axd=inset(0.115,Lb['pbot'],0.320,PANEL_H)
plot_pw(axd,-0.4,0.35,BLACK,lw=1.7,z=2)
xm=np.concatenate([np.linspace(-0.92,-0.45,7),np.linspace(-0.33,0.28,7),np.linspace(0.42,0.92,6)])
axd.plot(xm,utrue(xm)+rng.normal(0,0.06,xm.size),'o',color=RED,ms=3.2,zorder=3)
axd.set_ylim(0,1.15); axd.set_xlim(-1,1)
fig.text(0.115+0.320/2,Lb['xlab'],r'$x$',ha='center',va='top',fontsize=AL)
axp=inset(0.585,Lb['pbot'],0.340,PANEL_H)
def utrue_wig(x,j1,j2,amp,slope,bump):
    x=np.asarray(x,float); u=np.empty_like(x)
    m1=x<j1; m2=(x>=j1)&(x<j2); m3=x>=j2
    u[m1]=0.25*amp+0.18*np.sin(2.5*(x[m1]+1))+0.05*np.sin(5.0*x[m1])*bump
    u[m2]=0.95*amp-(0.22+slope)*(x[m2]-j1)+0.05*np.cos(4*x[m2])+0.05*np.sin(6*x[m2])*bump
    u[m3]=0.45*amp+0.30*(x[m3]-j2)**2+0.04*np.sin(5*x[m3])*bump
    return u
for k in range(8):
    j1=rng.normal(-0.4,0.055); j2=rng.normal(0.35,0.055)
    amp=rng.normal(1.0,0.09); slope=rng.normal(0.0,0.10); bump=rng.normal(0.0,1.0)
    for lo,hi in [(-1,j1),(j1,j2),(j2,1)]:
        xseg=np.linspace(lo,hi,250)
        axp.plot(xseg,utrue_wig(xseg,j1,j2,amp,slope,bump),color=GREEN,lw=0.7,zorder=2,alpha=0.55,solid_capstyle='round')
plot_pw(axp,-0.4,0.35,BLACK,lw=1.7,z=3)
axp.set_ylim(0,1.15); axp.set_xlim(-1,1)
fig.text(0.585+0.340/2,Lb['xlab'],r'$x$',ha='center',va='top',fontsize=AL)

fig.savefig('../out/figures/bayes.pdf',bbox_inches='tight',pad_inches=0.02)
print("saved")
