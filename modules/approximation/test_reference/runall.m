tic; dump('sectors.png','ref1.mat','GridSize',120,'ImageSize',64,'Verbose',false); toc
tic; dump('disk.png','ref2.mat','GridSize',100,'RationalDegree',150,'AnchorDegree',12,'Verbose',false); toc
tic; dump('sectors_p.png','ref3.mat','GridSize',[70 90],'ImageSize',48,'InnerIterations',8,'RungeShift',0.01i,'BoundaryColors',[0 0 0],'Verbose',false); toc
tic; dump('sectors.png','ref4.mat','GridSize',110,'ImageSize',50,'RationalDegree',900,'PlotBounds',[-2 2 -1.5 1.5],'MaxIterations',40,'NumBasins',3,'Verbose',false); toc
