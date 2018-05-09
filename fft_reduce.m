% --------------------------------------------------------------------
% Lehigh University - CSE
% CSE 326 - Machine Learning
% Komel Merchant
% -------------------------------------------------------------------
function fft_set =  fft_reduce(X, p_size)
    
    half_path = (p_size-1)/2;

    [N,D] = size(X);
    
    d = sqrt(D);
    
    fft_set = zeros(N,p_size.^2);
    
    
    
    
    for i  = 1:N 
        
        
        
   
       img = X(i,:);
       img = reshape(img,[d,d]);
       
       C_x = 16;
       C_y = 16;
       
       
       % ------------------
       % CONVERT TO FORIER DOMAIN
       % ------------------
       f_domain = fft2(double(img));
       f_domain = fftshift(f_domain); 
        
       f_domain = f_domain(C_x-half_path:C_x+half_path, C_y-half_path:C_y+half_path);
       
       [s,~] = size(f_domain);
       fft_set(i,:) = reshape(f_domain, [s.^2,1]);
        
        
    end






end