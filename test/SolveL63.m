Tspan = [0 50]
v0 = [1 0 0];
[tout,vout]=ode45('@L63',Tspan,v0);

%%% Plot in 3D
figure(1)
plot3(vout(:,1),vout(:,2),vout(:,3),'b')
hold on

v0 = [1+1E-5 0 0];
[tout,vout]=ode45('@L63',Tspan,v0);
plot3(vout(:,1),vout(:,2),vout(:,3),'r')

%%% Plot of (x,y,z) vs time 
figure(2)
plot(tout,vout(:,1),tout,vout(:,2),tout,vout(:,3));
hold on

v0 = [1 0 0];
[tout,vout]=ode45('@L63',Tspan,v0);
plot(tout,vout(:,1),tout,vout(:,2),tout,vout(:,3));

%(xp,yp,zp) are the solution with IC [1+1E-5 0 0]
legend('xp','yp','zp','x','y','z')

