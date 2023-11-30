%Initialization
n=40; %Dimension of state space
m=20; %Dimension of observation space
N=40; %Number of ensemble members
%H=eye(m,n); %Observation operator: 
H = zeros(m,n);
for k=1:m
%H(k,2*(k-1)+1)=1;
%H(k,2*(k-1)+2)=1;
H(k,k)=1;
end
H

localization_radius = 3;
C_localization = zeros( n );

for i=1:n
    for j=1:n
        C_localization( i, j ) = GC( abs( i - j ), localization_radius );
    end
end

dt = 0.1 %Time between observations
J = 1000; %Number of assimilation times
vt = zeros(n,J+1);
yt = zeros(m,J+1);
vt(1,1)=1;
Varr = zeros(n,N);
Vharr = zeros(n,N);

%Covariances
alpha=0.1;
beta =0.1;
C0 = beta^2*eye(n);
Sigma = beta^2*eye(n);
Gamma = alpha^2*eye(m);

%Get truth and synthetic observations
for j=1:J
Tspan = [(j)*dt,(j+1)*dt];
[tend,w]=ode45('@L96',Tspan,vt(:,j));
vt(:,j+1)=w(end,:)';
yt(:,j+1)=H*vt(:,j+1)+alpha*randn(1,m)'; 
end

figure(1)
plot3(vt(1,:),vt(2,:),vt(3,:));
title('Truth');

%ICs for ensembles
for k=1:N
Varr(:,k) = vt(:,1) + beta*randn(1,n)';
end

%Main Time Loop
for j=1:J
j
%Prediction of Ensembles
for k=1:N
Tspan = [(j)*dt,(j+1)*dt];
[tend,w]=ode45('@L96',Tspan,Varr(:,k));
Vharr(:,k)=w(end,:)' + beta*rand(1,n)';
end

%Sample mean
mhat = zeros(n,1);
for k=1:N
mhat = mhat + Vharr(:,k);
end
mhat = mhat/N;

%Sample covariance
Chat = zeros(n);
for k=1:N
covvec = Vharr(:,k)-mhat;
Chat = Chat + covvec*covvec';
end
Chat = Chat/(N-1);

% Apply the localization
Chat = C_localization.*Chat;

%Analysis
S = H*(Chat)*H'+Gamma;
for k=1:N
Innov(:,k) = yt(:,j+1)-H*Vharr(:,k)+alpha*randn(1,m)';
SinvI(:,k) = S\Innov(:,k);
end

KI = Chat*H'*SinvI;
Varr = Vharr+KI;
%No Analysis: comment the line below and uncomment the line above to add the Analysis
%Varr = Vharr;

%Sample mean of the Analysis!
mVarr = zeros(n,1);
for k=1:N
mVarr = mVarr + Varr(:,k);
end
mVarr = mVarr/N;

%Record RMSE
RMSE(j)=norm(mVarr - vt(:,j+1))/sqrt(n);

end

%Plot RMSE
figure(2)
plot([1:J],RMSE);
hold on
plot([1:J],alpha*ones(J,1),'g-');
title('RMSE');

meanVarr = Vharr;RMSE = mean(RMSE)

function gc_val = GC( dist, r )
    if ( dist >= 0 && dist < 1 )
        gc_val = 1 - ( ( 5 / 3 ) * dist ^ 2 ) + ( ( 5 / 8 ) * dist ^ 3 ) + ( ( 1 / 2 ) * dist ^ 4 ) - ( ( 1 / 4 ) * dist ^ 5 );

    elseif ( dist >= 1 && dist < r )
        gc_val = 4 - ( 5 * dist ) + ( ( 5 / 3 ) * dist ^ 2 ) + ( ( 5 / 8 ) * dist ^ 3 ) - ( ( 1 / 2 ) * dist ^ 4 ) - ( ( 1 / 12 ) * dist ^ 5 ) - ( 2 / ( 3 * dist ) );
    
    else
        gc_val = 0;
    end
    return
end
