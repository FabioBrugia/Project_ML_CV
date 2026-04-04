% Load_Plot_EEG_1
%
% Please acknowledge the use of this data in any publications by refering
% to the article: "Modeling the nonlinear cortical response in EEG evoked
% by wrist joint manipulation" by 
%  Vlaar et al., IEEE Trans Neural Syst Rehabil Eng 26:205-305, 2018
%
% The data is organized in a struct (data). The input (data.u) and output
% (data.y) are matrices with dimensions:  
%  [#participants, #realizations[M], #samples[N].
%
% The data has been averaged and downsampled (Section II-B). The data has
% been scaled/normalized and a time delay has been imposed (Section II-G)      
%
%
% The data may be used, copied, or redistributed as long as it is not sold
% and this copyright notice is reproduced on each copy made. The data is
% provided as is, without any express or implied warranties whatsoever. 
%
% On behalf of all authors, 
% Alfred C. Schouten
% Delft Laboratory for NeuroMechanics and Motor Control (NMC lab)
% Department of Biomechanical Engineering
% Delft University of Technology
%
% Mekelweg 2
% 2628 CD Delft
% The Netherlands
% e-mail: A.C.Schouten@tudelft.nl
% url:    www.3me.tudelft.nl/nmc
%
% February 20, 2019

%% load the data
close all
clear 
clc

load('Benchmark_EEG_small')

S = size(data.u,1); % #participants 
M = size(data.u,2); % #realizations
N = size(data.u,3); % #samples

fs = 256;           % sample frequency [Hz]
T = N/fs;           % segment length [s]
t=(0:N-1)'/fs;      % time vector

u=data.u;           % input, handle angle (normalized)
y=data.y;           % output, ICA component with highest SNR (normalized)

%% Fourier transform
U=fft(u,[],3);
Y=fft(y,[],3);
U=U(:,:,1:N/2+1);
Y=Y(:,:,1:N/2+1);
f=(0:fs/2)'/T;
% find frequencies with power
n=find(abs(squeeze(U(1,1,:)))>1e-0);

%% figure time-domain
h1=figure('Name','Input-output for 7 realizations (rows), for every participants one colored line');
for IdxM=1:M
    subplot(7,2,2*IdxM-1)
    plot(t,squeeze(u(:,IdxM,:))), box off
    subplot(7,2,2*IdxM)
    plot(t,squeeze(y(:,IdxM,:))), box off
end
subplot(721)
title('input')
subplot(722)
title('output')
subplot(7,2,13)
xlabel('time [s]')
subplot(7,2,14)
xlabel('time [s]')

%% figure frequency-domain
h2=figure('Name','Input-output for 7 realizations (rows), for every participants one colored line');
for IdxM=1:M
    subplot(7,2,2*IdxM-1)
    plot(f,squeeze(abs(U(:,IdxM,:)))),hold on,box off
    plot(f(n),transpose(squeeze(abs(U(:,IdxM,n)))),'o')
    xlim([0 50])
    subplot(7,2,2*IdxM)
    plot(f,squeeze(abs(Y(:,IdxM,:)))),hold on,box off
    plot(f(n),transpose(squeeze(abs(Y(:,IdxM,n)))),'o')
    xlim([0 50])
end
subplot(721)
title('input')
subplot(722)
title('output')
subplot(7,2,13)
xlabel('frequency [Hz]')
subplot(7,2,14)
xlabel('frequency [Hz]')
