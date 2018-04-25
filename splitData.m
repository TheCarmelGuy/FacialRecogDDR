%Function to split the data radomly accoding to a percentage
function [trainDataX,trainDataY,testDataX,testDataY] = splitData(X,Y,testPercent)
    
        %Obtain N of the data
        dataSize = size(X,1);
        
        %Obtain the desired % of the data for test
        DataPercent = round(testPercent*dataSize);
        
        %creates a random order of numbers from 1 to N
        idx = randperm(dataSize);
        
        %store x% of the total data as test
        testDataX = X(idx(1:DataPercent),:);
        testDataY = Y(idx(1:DataPercent));
        
        %store 100-x% of the total data as train
        trainDataX = X(idx(DataPercent+1:end),:);
        trainDataY = Y(idx(DataPercent+1:end));
        
end