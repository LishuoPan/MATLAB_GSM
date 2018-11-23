function [] = plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean_DCP,pVar_DCP,pMean_ADMM,figName)
figure(); hold on;
plot(xtrain,ytrain,'b','LineWidth',2);
f = [pMean_DCP(1:nTest) + 2*sqrt(pVar_DCP(1:nTest)); flip(pMean_DCP(1:nTest) - 2*sqrt(pVar_DCP(1:nTest)),1)];
fill([xtest(1:nTest); flip(xtest(1:nTest),1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
p1 = plot(xtest(1:nTest), ytest(1:nTest), 'g','LineWidth',2); 
p2 = plot(xtest(1:nTest), pMean_DCP(1:nTest),'b-','LineWidth',1.5);

p3 = plot(xtest(1:nTest), pMean_ADMM(1:nTest),'k-','LineWidth',1.5);

legend([p1, p2, p3],'test data','DCP','ADMM');

savefig(figName);
end