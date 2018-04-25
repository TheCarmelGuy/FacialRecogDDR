clc;
close;
clear;

t = cputime;

%load tha data
load ORL_32x32.mat

%define your k value
k = 3; %Note that K needs to be less than size(trainX,1)

%data split
[trainX,trainY,testX,testY] = splitData(fea,gnd,.20);

%note that we assume testX does not have labels and trainX has the labels
%we use testY to check the solution

%%  k-NN algorithm

    %Calculate the distances of every point for each of the test data points
    for i = 1:1:size(testX,1)
        for j = 1:1:size(trainX,1)
           dist(j) = EuclidianDistance(testX(i,:),trainX(j,:));
        end
        
        %at this point dist has all testX(i,:) distances to the train data
        %dist will be of dimentions [1Xsize(trainX,1)]
        
        %Find the k minimum distances
        for z = 1:1:k
            %calculate minimum distance of the dist array
            [value, index] = min(dist);
            
            %store minimum indeces in array
            minIndeces(z) = index;
            
            %eliminate value from dist by making it large so it does not go
            %into the min function
            dist(index) = inf;
        end
            
        %perform majority vote
            %Using the store indexes go get the train labels
            trainLabels = trainY(minIndeces);
            %Vote
            decidedLabel = mode(trainLabels);
        
        %store this in a descision array
        decidedLabels(i) = decidedLabel;
        
    end
    
    %get the accuracy
    %check the results
    diff = decidedLabels' - testY;
    
    %count the zeros in the array
    correctLabel = nnz(~diff);
    
    %check accuracy
    accuracy = correctLabel./size(testX,1);
    
    fprintf('ACCURACY OF KNN \n');
    fprintf('%f \n\n',accuracy);
    
    e = cputime-t;
    
    fprintf('COMPUTATIONAL TIME (SECONDS) \n');
    fprintf('%f \n',e);
    
    
%% Helper functions

%euclidian distance
function [Distance] = EuclidianDistance(A,B)
        Distance = sqrt(((A - B)*((A - B)')));
end

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