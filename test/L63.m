function [xdot] = L63(~,x)
sigma = 10;
rho = 28;
beta = 8/3;
xdot = [sigma*x(2,:) - sigma*x(1,:);...
        x(1,:)*rho - x(1,:).*x(3,:)-x(2,:);...
        x(1,:).*x(2,:) - beta*x(3,:)];
end   
