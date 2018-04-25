

function feature_vectors = harris_features(k, threshold, X, patch_size) 

    [rows, columns] = size(Ig); 
    

    %compute dirivatives with respect to both directions 
    [Ix, Iy] = imgradientxy(Ig);

    Ix2 = Ix.*Ix;
    Iy2 = Iy.*Iy;
    Ixy = Ix.*Iy;

    %Convolve three images with a gaussian

    smooth_xx = conv2(gaussian,Ix2);
    smooth_yy = conv2(gaussian,Iy2);
    smooth_xy = conv2(gaussian,Ixy);   

    i_mask = zeros(rows, columns);

    for i = 1:rows  
        for j = 1:columns

            %Get matix M for every pixel 
            M = [smooth_xx(i,j) ,smooth_xy(i,j);
                smooth_xy(i,j) , smooth_yy(i,j)];

            [e_vectors,e_vals] = eig(M);

            R =  e_vals(1,1)*e_vals(2,2) - k*((e_vals(1,1)+e_vals(2,2))).^2;

            %Thresholding for a corner
            if(R > threshold)
                 i_mask(i,j) = 255;
            end


        end 
    end
    
    % aggregate all local minima 
    non_max_out = i_mask > imdilate(i_mask, [1 1 1; 1 0 1; 1 1 1]);
    
    [x , y] = find(non_max_out);
    [feature_length,~] =  size(x);
    
    features_vector = zeros(feature_length, patch_size, patch_size);
    
    
    
    
    
    
    
    

end

