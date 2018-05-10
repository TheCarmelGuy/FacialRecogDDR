% --------------------------------------------------------------------
% Lehigh University - CSE
% CSE 326 - Machine Learning
% Gustavo Grinsteins
% --------------------------------------------------------------------

%Variables
%Hidden Nodes: { 10, 36, 100}
%PCA dimentions	K: {30, 50,  70,  200, 300}
%W:{10^2, 20^2, 30^2}

clear;
clc;

time2 = cputime;
hidden10Error = 0;
hidden36Error = 0;
hidden100Error = 0;
hidden10Time = 0;
hidden36Time = 0;
hidden100Time = 0;

for i = 1:1:9
    clearvars -except hidden10Error hidden36Error hidden100Error hidden10Time hidden36Time hidden100Time i time2;
    
    %declare the hidden nodes
    if i <= 3
        hiddenNodes = 10;
    elseif i > 3 && i <= 6
        hiddenNodes = 36;
    elseif i > 6 && i <= 9
        hiddenNodes = 100; 
    end
    
    time = cputime;
    
    load ORL_32x32.mat
    
    numLabels = 40;
    
    %only for multiple output
    Y = arrangeY(gnd, numLabels); %array of zeros with a single 1
    
    %change this value
    %k = 30;
    
    %test PCA
    %[Z,vecs,vals] = pca(fea,k);
    
    %load the fourier transform data
    %load fft_10.mat
    %load fft_20.mat
    load fft_30.mat
    
    %split the data
    [trainDataX,trainDataY,testDataX,testDataY] = splitDataNN(thirty,Y,.10);
    
    %Do with pca and without pca
    x = trainDataX';
    t = trainDataY';
    
    % Choose a Training Function
    trainFcn = 'traingd';  %gradient descent backpropagation.
    
    % Create a Pattern Recognition Network
    hiddenLayerSize = hiddenNodes;
    net = patternnet(hiddenLayerSize, trainFcn);
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideParam.trainRatio = 90/100;
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 10/100;
    
    %learning rate
    net.trainParam.lr = 1;
    
    %suppress the GUI
    net.trainParam.showWindow = false;
    
    % Train the Network
    [net,tr] = train(net,x,t);
    
    %test
    x = testDataX';
    t = testDataY';
    
    % Test the Network
    y = net(x);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    e = cputime-time;
    
    %average Values
    if i <= 3
        hidden10Error = hidden10Error + percentErrors;
        hidden10Time = hidden10Time + e;
    elseif i > 3 && i <= 6
        hidden36Error = hidden36Error + percentErrors;
        hidden36Time = hidden36Time + e;
    elseif i > 6 && i <= 9
        hidden100Error = hidden100Error + percentErrors;
        hidden100Time = hidden100Time + e;
    end
    
    
    
end

hidden10Error = hidden10Error/3;
hidden36Error = hidden36Error/3;
hidden100Error = hidden100Error/3;
hidden10Time = hidden10Time/3;
hidden36Time = hidden36Time/3;
hidden100Time = hidden100Time/3;

for i = 1:1:3

    if i == 1
        %declare the hidden nodes
        fprintf('Hidde Nodes: 10 \n')
        fprintf('Accuracy: %.3f \n',1-hidden10Error)
        fprintf('CPU time: %.3f \n', hidden10Time)
    elseif i == 2
        %declare the hidden nodes
        fprintf('Hidde Nodes: 36 \n')
        fprintf('Accuracy: %.3f \n',1-hidden36Error)
        fprintf('CPU time: %.3f \n', hidden36Time)
    elseif i == 3
        %declare the hidden nodes
        fprintf('Hidde Nodes: 100 \n')
        fprintf('Accuracy: %.3f \n',1-hidden100Error)
        fprintf('CPU time: %.3f \n', hidden100Time)
    end

end 

fprintf('Total time (minutes): %.3f \n', (cputime - time2)/60)


%the purpose of this function is to get the labels into appropriate format
function [Y] = arrangeY(gnd, numLabels)


%initialize the length of the array
Y = zeros(length(gnd),40);

for i = 1:1:length(gnd)
    for j = 1:1:numLabels
        if j == gnd(i)
            
            Y(i,j) = 1;
            
        end
    end
end

end

%get a training set
function [trainDataX,trainDataY,testDataX,testDataY] = splitDataNN(X,Y,testPercent)

%Obtain N of the data
dataSize = size(X,1);

%Obtain the desired % of the data for test
DataPercent = round(testPercent*dataSize);

%creates a random order of numbers from 1 to N
idx = randperm(dataSize);

%store x% of the total data as test
testDataX = X(idx(1:DataPercent),:);
testDataY = Y(idx(1:DataPercent),:);

%store 100-x% of the total data as train
trainDataX = X(idx(DataPercent+1:end),:);
trainDataY = Y(idx(DataPercent+1:end),:);

end
