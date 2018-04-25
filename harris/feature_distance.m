

%% Distance Metric 
% L1 minimum sum of distances
%
%
function dscore = feature_distance(x_1, x_2)
    
     [N, x, y] = size(x_1);

     for i = 1:N
         flatten_x1(i) = sum(x_1(i,:));
         flatten_x2(i) = sum(x_2(i,:));
   
     
     end
     flatten_x1 = sort(flatten_x1);
     flatten_x2 = sort(flatten_x2);
    
     dscore = sum(abs(flatten_x1-flatten_x2)); 




end