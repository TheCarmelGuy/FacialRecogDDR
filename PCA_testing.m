clc; clear all;

load Data/ORL_32x32

%30,50,70,200,300
[pca30_Z , pca30_ves , pca30_vals ] = pca(fea,30 );
[pca50_Z , pca50_ves , pca50_vals ] = pca(fea,50 );
[pca70_Z , pca70_ves , pca70_vals ] = pca(fea,70 );
[pca200_Z, pca200_ves, pca200_vals] = pca(fea,200);
[pca300_Z, pca300_ves, pca300_vals] = pca(fea,300);

[trainx   , trainy   , testx   , testy   ] = splitData(fea     , gnd, .2);
[train30x , train30y , test30x , test30y ] = splitData(pca30_Z , gnd, .2);
[train50x , train50y , test50x , test50y ] = splitData(pca50_Z , gnd, .2);
[train70x , train70y , test70x , test70y ] = splitData(pca70_Z , gnd, .2);
[train200x, train200y, test200x, test200y] = splitData(pca200_Z, gnd, .2);
[train300x, train300y, test300x, test300y] = splitData(pca300_Z, gnd, .2);

%k = 1, 2, 5, 10

%For k = 1
[e_f_1 ,   accuracyf_1] = KNN(testx   , testy   , trainx   , trainy   , 1);
[e30_1 ,  accuracy30_1] = KNN(test30x , test30y , train30x , train30y , 1);
[e50_1 ,  accuracy50_1] = KNN(test50x , test50y , train50x , train50y , 1);
[e70_1 ,  accuracy70_1] = KNN(test70x , test70y , train70x , train70y , 1);
[e200_1, accuracy200_1] = KNN(test200x, test200y, train200x, train200y, 1);
[e300_1, accuracy300_1] = KNN(test300x, test300y, train300x, train300y, 1);

%For k = 2
[ e_f_2,   accuracyf_2] = KNN(testx   , testy   , trainx   , trainy   , 2);
[ e30_2,  accuracy30_2] = KNN(test30x , test30y , train30x , train30y , 2);
[ e50_2,  accuracy50_2] = KNN(test50x , test50y , train50x , train50y , 2);
[ e70_2,  accuracy70_2] = KNN(test70x , test70y , train70x , train70y , 2);
[e200_2, accuracy200_2] = KNN(test200x, test200y, train200x, train200y, 2);
[e300_2, accuracy300_2] = KNN(test300x, test300y, train300x, train300y, 2);

%For k = 5
[ e_f_5,   accuracyf_5] = KNN(testx   , testy   , trainx   , trainy   , 5);
[ e30_5,  accuracy30_5] = KNN(test30x , test30y , train30x , train30y , 5);
[ e50_5,  accuracy50_5] = KNN(test50x , test50y , train50x , train50y , 5);
[ e70_5,  accuracy70_5] = KNN(test70x , test70y , train70x , train70y , 5);
[e200_5, accuracy200_5] = KNN(test200x, test200y, train200x, train200y, 5);
[e300_5, accuracy300_5] = KNN(test300x, test300y, train300x, train300y, 5);

%For k = 10
[ e_f_10,   accuracyf_10] = KNN(testx   , testy   , trainx   , trainy   , 10);
[ e30_10,  accuracy30_10] = KNN(test30x , test30y , train30x , train30y , 10);
[ e50_10,  accuracy50_10] = KNN(test50x , test50y , train50x , train50y , 10);
[ e70_10,  accuracy70_10] = KNN(test70x , test70y , train70x , train70y , 10);
[e200_10, accuracy200_10] = KNN(test200x, test200y, train200x, train200y, 10);
[e300_10, accuracy300_10] = KNN(test300x, test300y, train300x, train300y, 10);


figure(1)
set(gcf, 'Position', [100,100,1000,800]);
c = categorical({'Not reduced: 1024','M = 30','M = 50','M = 70', 'M= 200', 'M = 300'});
c = reordercats(c,{'Not reduced: 1024','M = 30','M = 50','M = 70', 'M= 200', 'M = 300'});
Y = [accuracyf_1	accuracyf_2	    accuracyf_5    accuracyf_10
     accuracy30_1	accuracy30_2	accuracy30_5   accuracy30_10
     accuracy50_1	accuracy50_2	accuracy50_5   accuracy50_10 
     accuracy70_1	accuracy70_2	accuracy70_5   accuracy70_10
     accuracy200_1	accuracy200_2	accuracy200_5  accuracy200_10
     accuracy300_1	accuracy300_2	accuracy300_5  accuracy300_10];
h = bar(c,Y);
title('Comparison of KNN Accuracy by K Value and Data Size');
ylabel('%Accuracy')
xlabel('Data Size')
l = cell(1,3);
l{1}='k = 1'; l{2}='k = 2'; l{3}='k = 5'; l{4} = 'k = 10';   
grid on
lgd = legend(h,l);
title(lgd,'K Value')

figure(2)
set(gcf, 'Position', [100,100,1000,800]);
c = categorical({'Not reduced: 1024','M = 30','M = 50','M = 70', 'M= 200', 'M = 300'});
c = reordercats(c,{'Not reduced: 1024','M = 30','M = 50','M = 70', 'M= 200', 'M = 300'});
Y = [e_f_1	e_f_2	e_f_5   e_f_10
     e30_1	e30_2	e30_5   e30_10
     e50_1	e50_2	e50_5   e50_10 
     e70_1	e70_2	e70_5   e70_10
     e200_1	e200_2	e200_5  e200_10
     e300_1	e300_2	e300_5  e300_10];
h = bar(c,Y);
title('Comparison of KNN Computation Time by K Value and Data Size')
ylabel('%Computation Time')
xlabel('Data Size')
l = cell(1,3);
l{1}='k = 1'; l{2}='k = 2'; l{3}='k = 5'; l{4} = 'k = 10';   
grid on
lgd = legend(h,l);
title(lgd,'K Value')

%euclidian distance
function [Distance] = EuclidianDistance(A,B)
        Distance = sqrt(((A - B)*((A - B)')));
end

function [e, accuracy] = KNN(testX, testY, trainX, trainY, k)

    t = cputime;

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
    
    e = cputime-t;
    
    fprintf('FOR SIZE: %d AND K: %d\n------------------------\n',size(testX,2),k);
    fprintf('ACCURACY OF KNN \n');
    fprintf('%f \n\n',accuracy);
    fprintf('COMPUTATIONAL TIME (SECONDS) \n');
    fprintf('%f \n\n\n',e);
end