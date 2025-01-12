clearvars -except Fs ;
load('binary_tester_V1.mat')
close all;clc
Fs=44100;

%fetch audio data
rec=audiorecorder(Fs,16,1);
disp ("Recording Started")
recordblocking(rec,5)
disp("Recording Stopped")
data=getaudiodata(rec);

%making sure data is 1D row vector
if size(data,1)>1 & size(data,2)>=1
   data=data(:,1);%making sure only one channel is getting into data
   data=data';
end

%plotting the input signal
t=(0:length(data)-1)/Fs;
figure;
plot(t,data)
xlabel('Time (seconds)')
ylabel('Amplitude')

%check fourier Transform of signal
L = length(data);
NFFT = 2^nextpow2(L);
y_fft = abs(fft(data,NFFT));
% creating frequency axis
freq = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure;
plot(freq,y_fft(1:NFFT/2+1));
title('Single-Sided Amplitude Spectrum of y(t)');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

%design filter and do filtering if needed
fc = 10^4; % Cutoff frequency
o = 5; % Order
wn = fc / (Fs/2); % Normalized cutoff frequency
[b,a] = butter(o, wn, 'low');

% Filter the signal
data_filtered= filter(b,a,data);
data=data_filtered;

%segmenting
%output is in segments variable

segment_duration=2.5; %seconds
len_segment = fix((Fs * segment_duration)+1);
num_segments = ceil(length(data) / len_segment);
ham_win=(hamming(len_segment))';    
% Initializing segments matrix
segments = zeros(num_segments, len_segment);
    
for i = 1:num_segments
   % Determine the start and end indices for the current segment
   start_idx = (i - 1) * len_segment + 1;
   end_idx = min(start_idx + len_segment - 1, length(data));
 
   segment = data(start_idx:end_idx);
        
  if length(segment) < len_segment
      segment = [segment, zeros(1, len_segment - length(segment))]; %zero padding if needed
  end
  
  segment=segment .* ham_win; %comment out if unnecessary
  segment = (segment - mean(segment)) / std(segment); %normalizing
  segments(i, :) = segment;
end

clearvars -except segments Fs features labels mdl
features=find_features(segments,Fs);

pdct=predict(mdl,features); % it has column vector of predicted labels
 if nnz(pdct)>length(pdct)
     disp("Patient have Parkinson Symptomps")
 else
     disp("Person is normal")
 end



function feat_matrix=find_features(segments,Fs)

[r,c] = size(segments);
pitch = zeros(r,1);
for i = 1:r
    % Compute the autocorrelation function
    temp = segments(i,:);
    autocorr_func = xcorr(temp);
% Find the peaks in the autocorrelation function
   [peaks, peak_locs] = findpeaks(autocorr_func);
% Find the peak with the highest amplitude (excluding the first peak, which corresponds to zero lag)
   [max_peak, max_peak_idx] = max(peaks(2:end));
% Compute the lag corresponding to the peak
   lag = peak_locs(max_peak_idx + 1) - peak_locs(1); % Add 1 to account for excluding the first peak
% Compute the fundamental frequency (F0) in Hz
   F0 = Fs / lag;
   pitch(i) = F0;
end

format long
Tn = 1./pitch;
jitt = zeros(r,1);
for i = 1:r-1
    temp = (1/r-1)*(Tn(i) - Tn(i+1))*100; %expressed as a percentage
    jitt(i+1) = temp;
end
jitt = abs(jitt);

shim = zeros(r,1);
temp = 0;
temp_shim = zeros(r,1);
for i = 1:r
    temp = segments(i,:);
    for j = 1:c-1
        temp_shim(j+1) = (temp(j) - temp(j+1))/temp(j);
        temp_shim(j+1) = abs(temp_shim(j+1));
    end
    sump = cumsum(temp_shim);
    sump = sump(end);
    
    shim(i) = (1/c-1)*sump*100;
end
shim = abs(shim);

xa = zeros(1,c);
for i = 1:c
    xa(i) = mean(segments(:,i));
end
xa_sqr = xa.^2;
H = r*cumsum(xa_sqr);
H = H(end); % harmonic component
noise_comp = 0;
for i = 1:r
    temp = segments(i,:) - xa;
    temp =  temp.^2;
    temp = cumsum(temp);
    noise_comp = noise_comp + temp(end);
end
N = noise_comp;
HNR = 10 * log10(H/N) + zeros(r,1);
feat_matrix=[pitch,jitt,shim,HNR];
%making the label matrix
%label=1; %0 for normal & 1 for parkinson
end
