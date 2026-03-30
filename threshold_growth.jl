# Plots 1D threshold and growth rate for BEC with single mirror feedback

using PyPlot

points=100000

Delta=1000
b0=100
R=0.99


m=133*1.67e-27
hbar=6.626e-34/2/pi
gam=2*pi*5.22e6
c=3e8
wavelength=852e-9
k0=2*pi/wavelength
kb=1.34e-23
lambda_c=5e-6


q_c=2.0*pi/lambda_c
omega_r = hbar*q_c^2/2/m/gam        #omega_r / Gamma
#print ('omega_r=',omega_r)

#print const

q2=collect(range(1.0e-3,11.0,points)) # q_bar^2

thresh=2.0*omega_r/(b0*R)*q2./sin.(q2*pi/2.0)  #Threshold p0

disc_ind=(thresh.>90.0*omega_r/(b0*R))
x_disc=q2[disc_ind]
thresh[disc_ind].=NaN

p0=3e-7
p0_over_thresh=(p0./thresh.-1.0).*(p0.>thresh).*(thresh.!==NaN).*(thresh.>1.0e-15)
g=omega_r*q2.*sqrt.(p0_over_thresh)


ion()

figure(1)
subplot(211)
#plot(xvec,thresh,'k-',xvec,thresh_c,'r--')      #plotting
plot(q2,thresh,"k-")
#plot(xp1,threshp1,"k-",xp2,threshp2,"w-",xp3,threshp3,"k-",xp4,threshp4,"w-",xp5,threshp5,"k-",xp6,threshp6,"w-")
#plt.plot(xm6, thresh_cm6,'r--', xm5,thresh_cm5,'w-', xm4,thresh_cm4,'r--', xm3,thresh_cm3,'w-', xm2,thresh_cm2,'r--', xm1,thresh_cm1,'w-', xp1,thresh_cp1,'r--', xp2,thresh_cp2,'w-',xp3,thresh_cp3,'r--',xp4,thresh_cp4,'w-',xp5,thresh_cp5,'r--',xp6,thresh_cp6,'w-')

#plot(xvec,thresh_c,'r.')  
axhline(y=p0,color="b",ls="dotted")
#plot(pi/2*1/25,p0,'ko',pi/2*4/25,p0,'ko',pi/2*9/25,p0,'ko',pi/2*16/25,p0,'ko',pi/2,p0,'ko',pi/2*36/25,p0,'ko')

#axvline(x=0,color='g',ls='dashdot',lw=1)
#axvline(x=pi,color='g',ls='dashdot',lw=1)
#axvline(x=2*pi,color='g',ls='dashdot',lw=1)
#axvline(x=3*pi,color='g',ls='dashdot',lw=1)

text(10,30.0*omega_r/(b0*R),"(a)",fontsize=18)
xlim([minimum(q2),maximum(q2)])
ylim([0,40.0*omega_r/(b0*R)])
#xlabel(r'$q^2 k_0/d$',fontsize=18)
ylabel("\$p_{th}\$",fontsize=18)

subplot(212)
plt.plot(q2,g,"k-")
#plot(xvec,g_c,'r--')
#plot(pi/2,2.82,'ko')

xlim([minimum(q2),maximum(q2)])
ylim([0,maximum(g)*1.1])
text(10,maximum(g)*0.8,"(b)",fontsize=18)
xlabel("\$\\bar{q}^2\$",fontsize=18)
ylabel("\$\\bar{g}\$",fontsize=18)
tight_layout()

savefig("thresh_growth.pdf",dpi=300)
savefig("thresh_growth.png",dpi=300)

