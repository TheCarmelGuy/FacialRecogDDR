% --------------------------------------------------------------------
% Lehigh University - CSE
% CSE 326 - Machine Learning
% Gustavo Grinsteins
% --------------------------------------------------------------------
%histogram test
clear;
clc;

%create a bar plot
figure(1)
subplot(2,1,1)
c = categorical({'No FFT','w = 30x30','w = 20x20','k = 10x10'});
c = reordercats(c,{'No FFT','w = 30x30','w = 20x20','k = 10x10'});
Y = [0.75	0.93	0.917
0.475	0.725	0.683
0.508	0.792	0.733
0.408	0.75	0.808];
h = bar(c,Y);
ylabel('%Accuracy')
xlabel('Fast Fourier Transforms')
l = cell(1,3);
l{1}='10'; l{2}='36'; l{3}='100';   
grid on
lgd = legend(h,l);
title(lgd,'Hidden Nodes')

%create a bar plot
subplot(2,1,2)
c = categorical({'No FFT','w = 30x30','w = 20x20','k = 10x10'});
c = reordercats(c,{'No FFT','w = 30x30','w = 20x20','k = 10x10'});
Y = [9.403	11.9	17.5
8.463	10.9	16.4
5.9	8.74	11.39
4.167	6.037	8.27];
h = bar(c,Y);
ylabel('Computational Time (seconds)')
xlabel('Fast Fourier Transforms')
l = cell(1,3);
l{1}='10'; l{2}='36'; l{3}='100';   
grid on
lgd = legend(h,l);
title(lgd,'Hidden Nodes')
suptitle({'Pattern Recognition Neural Network Performance Vs FFT', 'Cross-Entropy loss function','Softmax output layer', 'Gradient Descent Backpropagation', '\eta = 1'})


