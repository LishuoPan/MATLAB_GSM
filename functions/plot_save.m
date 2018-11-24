function [] = plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName)
figure(); hold on;
plot(xtrain,ytrain,'b','LineWidth',2);
f = [pMean(1:nTest) + 2*sqrt(pVar(1:nTest)); flip(pMean(1:nTest) - 2*sqrt(pVar(1:nTest)),1)];
fill([xtest(1:nTest); flip(xtest(1:nTest),1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
plot(xtest(1:nTest), ytest(1:nTest), 'g','LineWidth',2); 
plot(xtest(1:nTest), pMean(1:nTest), 'k','LineWidth',2);

savefig(figName);
end