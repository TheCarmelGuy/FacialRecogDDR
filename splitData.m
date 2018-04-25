
function [trainDataX,trainDataY,testDataX,testDataY] = splitData(X,Y,testPercent)

    dataSize = size(X,1);
    DataPercent = round(testPercent*dataSize);
    idx = randperm(dataSize);
    indexToTest = (idx<=DataPercent);
    indexToTrain = (idx>DataPercent);

    trainDataX = X(indexToTrain,:);
    trainDataY = Y(indexToTrain);

    testDataX = X(indexToTest,:);
    testDataY = Y(indexToTest);

end