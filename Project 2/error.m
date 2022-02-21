a=(1/(2*pi*sqrt(1-(0.75^2))));

fun= @(x,y) exp(-0.5*(x.^2-(2*0.75*(x.*y))+(y.^2)));
xmin=-5;
xmax=0;
ymin=-5;
ymax=0;

q1 = integral2(fun,xmin,xmax,ymin,ymax);

fun2= @(x,y) exp(-0.5*(x.^2-(2*0.75*(x.*y))+(y.^2)));
xmin2=0;
xmax2=5;
ymin2=0;
ymax2=5;

q2 = integral2(fun2,xmin2,xmax2,ymin2,ymax2);

fun3= @(x,y) exp(-0.5*(x.^2+(2*0.75*(x.*y))+(y.^2)));
xmin3=0;
xmax3=5;
ymin3=-5;
ymax3=0;

q3 = integral2(fun3,xmin3,xmax3,ymin3,ymax3);

fun4= @(x,y) exp(-0.5*(x.^2+(2*0.75*(x.*y))+(y.^2)));
xmin4=-5;
xmax4=0;
ymin4=0;
ymax4=5;

q4 = integral2(fun4,xmin4,xmax4,ymin4,ymax4);

Error=0.5*a*(q1+q2+q3+q4)