% Simulation Plot
clc;clear;

load('Elec_Pruned0_PruneRate02_SubIter1000_rho100_50_Epsilon15_MaxIL1000_Results.mat');
alphaPrune0 = alpha;
AugObjPrune0 = AugObjEval;
OriObjPrune0 = OriObjEval;
TestMSEPrune0 = TestMSEList;
TrainMSEPrune0 = TrainMSEList;
TimePrune0 = TimeList;
SSubIterPrune0 = SSubIterList;
GapPrune0 = Gap;

load('Elec_Pruned3_PruneRate02_SubIter1000_rho100_50_Epsilon15_MaxIL1000_Results.mat');
alphaPrune3 = alpha;
AugObjPrune3 = AugObjEval;
OriObjPrune3 = OriObjEval;
TestMSEPrune3 = TestMSEList;
TrainMSEPrune3 = TrainMSEList;
TimePrune3 = TimeList;
SSubIterPrune3 = SSubIterList;
GapPrune3 = Gap;


load('Elec_Pruned5_PruneRate02_SubIter1000_rho100_50_Epsilon15_MaxIL1000_Results.mat');
alphaPrune5 = alpha;
AugObjPrune5 = AugObjEval;
OriObjPrune5 = OriObjEval;
TestMSEPrune5 = TestMSEList;
TrainMSEPrune5 = TrainMSEList;
TimePrune5 = TimeList;
SSubIterPrune5 = SSubIterList;
GapPrune5 = Gap;

load('Elec_Pruned7_PruneRate02_SubIter1000_rho100_50_Epsilon15_MaxIL1000_Results.mat');
alphaPrune7 = alpha;
AugObjPrune7 = AugObjEval;
OriObjPrune7 = OriObjEval;
TestMSEPrune7 = TestMSEList;
TrainMSEPrune7 = TrainMSEList;
TimePrune7 = TimeList;
SSubIterPrune7 = SSubIterList;
GapPrune7 = Gap;


% Performance plot
figure;
hold on;
plot(TrainMSEPrune0, 'DisplayName', 'No Pruning');
plot(TrainMSEPrune3, 'DisplayName', '48.8% Pruning');
plot(TrainMSEPrune5, 'DisplayName', '67.2% Pruning');
plot(TrainMSEPrune7, 'DisplayName', '79.0% Pruning');
legend('show');

