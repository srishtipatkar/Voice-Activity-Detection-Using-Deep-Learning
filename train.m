clc
clear
adsTrain = audioDatastore('google_speech/train', "Includesubfolders",true);
adsValidation = audioDatastore('google_speech\validation', "Includesubfolders",true);
[data,adsInfo] = read(adsTrain);
Fs = adsInfo.SampleRate;
win = hamming(50e-3 * Fs,'periodic');
reset(adsTrain)
adsTrain = shuffle(adsTrain);
adsValidation = shuffle(adsValidation);
TM = [];
for index1 = 1:500
     data = read(adsTrain); 
     [~,T] = detectSpeech(data,Fs,'Window',win);
     TM = [TM;T];
end
T = mean(TM);
reset(adsTrain)
duration = 6000*Fs;
audioTraining = zeros(duration,1);
maskTraining = zeros(duration,1);
maxSilenceSegment = 2;
numSamples = 1;    

while numSamples < duration
    data = read(adsTrain);
    data = data ./ max(abs(data)); % Normalize amplitude

    % Determine regions of speech
    idx = detectSpeech(data,Fs,'Window',win,'Thresholds',T);

    % If a region of speech is detected
    if ~isempty(idx)
        
        % Extend the indices by five frames
        idx(1,1) = max(1,idx(1,1) - 5*numel(win));
        idx(1,2) = min(length(data),idx(1,2) + 5*numel(win));
        
        % Isolate the speech
        data = data(idx(1,1):idx(1,2));
        
        % Write speech segment to training signal
        audioTraining(numSamples:numSamples+numel(data)-1) = data;
        
        % Set VAD baseline
        maskTraining(numSamples:numSamples+numel(data)-1) = true;
        
        % Random silence period
        numSilenceSamples = randi(maxSilenceSegment*Fs,1,1);
        numSamples = numSamples + numel(data) + numSilenceSamples;
    end
end
[noise,f] = audioread("longNoise_mono16.mp3");
noise = resample(noise,2,1);
audioTraining = audioTraining(1:numel(noise));
SNR = -10;
noise = 10^(-SNR/20) * noise * norm(audioTraining) / norm(noise);
audioTrainingNoisy = audioTraining + noise; 
audioTrainingNoisy = audioTrainingNoisy / max(abs(audioTrainingNoisy));
speechIndices = detectSpeech(audioTrainingNoisy,Fs,'Window',win);

speechIndices(:,1) = max(1,speechIndices(:,1) - 5*numel(win));
speechIndices(:,2) = min(numel(audioTrainingNoisy),speechIndices(:,2) + 5*numel(win));

noisyMask = zeros(size(audioTrainingNoisy));
for ii = 1:size(speechIndices)
    noisyMask(speechIndices(ii,1):speechIndices(ii,2)) = 1;
end
duration = 700*Fs;
audioValidation = zeros(duration,1);
maskValidation = zeros(duration,1);
numSamples = 1;    
while numSamples < duration
    data = read(adsValidation);
    data = data ./ max(abs(data)); % Normalize amplitude
    
    % Determine regions of speech
    idx = detectSpeech(data,Fs,'Window',win,'Thresholds',T);
    
    % If a region of speech is detected
    if ~isempty(idx)
        
        % Extend the indices by five frames
        idx(1,1) = max(1,idx(1,1) - 5*numel(win));
        idx(1,2) = min(length(data),idx(1,2) + 5*numel(win));

        % Isolate the speech
        data = data(idx(1,1):idx(1,2));
        
        % Write speech segment to training signal
        audioValidation(numSamples:numSamples+numel(data)-1) = data;
        
        % Set VAD Baseline
        maskValidation(numSamples:numSamples+numel(data)-1) = true;
        
        % Random silence period
        numSilenceSamples = randi(maxSilenceSegment*Fs,1,1);
        numSamples = numSamples + numel(data) + numSilenceSamples;
    end
end
noise = audioread("longNoise_mono16.mp3");
noise = resample(noise,2,1);
noise = noise(1:duration);
audioValidation = audioValidation(1:numel(noise));
noise = 10^(-SNR/20) * noise * norm(audioValidation) / norm(noise);
audioValidationNoisy = audioValidation + noise; 
audioValidationNoisy = audioValidationNoisy / max(abs(audioValidationNoisy));

afe = audioFeatureExtractor('SampleRate',Fs, ...
    'Window',hann(256,"Periodic"), ...
    'OverlapLength',128, ...
    ...
    'spectralCentroid',true, ...
    'spectralCrest',true, ...
    'spectralEntropy',true, ...
    'spectralFlux',true, ...
    'spectralKurtosis',true, ...
    'spectralRolloffPoint',true, ...
    'spectralSkewness',true, ...
    'spectralSlope',true, ...
    'harmonicRatio',true);

numsam=1;
FT=[];
frame1=5*60*Fs;
while numsam+frame1<numel(audioTrainingNoisy)
    featuresTraining = extract(afe,audioTrainingNoisy(numsam:numsam+frame1-1));
    FT=[FT;featuresTraining];
    numsam=numsam+frame1;
end
M = mean(FT,1);
S = std(FT,[],1);
FT = (FT - M) ./ S;

% featuresTraining = extract(afe,audioTrainingNoisy);
% [numWindows,numFeatures] = size(featuresTraining);
% M = mean(featuresTraining,1);
% S = std(featuresTraining,[],1);
% % featuresTraining = (featuresTraining - M) ./ S;
% featuresValidation = extract(afe,audioValidationNoisy);
% featuresValidation = (featuresValidation - mean(featuresValidation,1)) ./ std(featuresValidation,[],1);

numsam=1;
FV=[];
while numsam+frame1<numel(audioValidationNoisy)
    featuresValidation = extract(afe,audioValidationNoisy(numsam:numsam+frame1-1));
    FV=[FV;featuresValidation];
    numsam=numsam+frame1;
end

FV = (FV - mean(FV,1)) ./ std(FV,[],1);

windowLength = numel(afe.Window);
hopLength = windowLength - afe.OverlapLength;
range = (hopLength) * (1:size(FT,1)) + hopLength;
maskMode = zeros(size(range));
for index = 1:numel(range)
    maskMode(index) = mode(maskTraining( (index-1)*hopLength+1:(index-1)*hopLength+windowLength ));
end
maskTraining = maskMode.';

maskTrainingCat = categorical(maskTraining);
range = (hopLength) * (1:size(FV,1)) + hopLength;
maskMode = zeros(size(range));
for index = 1:numel(range)
    maskMode(index) = mode(maskValidation( (index-1)*hopLength+1:(index-1)*hopLength+windowLength ));
end
maskValidation = maskMode.';

maskValidationCat = categorical(maskValidation);
sequenceLength = 150;
sequenceOverlap = round(0.75*sequenceLength);

trainFeatureCell = helperFeatureVector2Sequence(FT',sequenceLength,sequenceOverlap);
trainLabelCell = helperFeatureVector2Sequence(maskTrainingCat',sequenceLength,sequenceOverlap);
layers = [ ...    
    sequenceInputLayer( size(FV,2) )    
    bilstmLayer(200,"OutputMode","sequence")   
    bilstmLayer(200,"OutputMode","sequence")   
    fullyConnectedLayer(2)   
    softmaxLayer   
    classificationLayer      
    ];
maxEpochs = 20;
miniBatchSize = 64;
options = trainingOptions("adam", ...
    "MaxEpochs",maxEpochs, ...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Verbose",0, ...
    "SequenceLength",sequenceLength, ...
    "ValidationFrequency",floor(numel(trainFeatureCell)/miniBatchSize), ...
    "ValidationData",{FV.',maskValidationCat.'}, ...
    "Plots","training-progress", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",5);
doTraining = true;
if doTraining
   [speechDetectNet,netInfo] = trainNetwork(trainFeatureCell,trainLabelCell,layers,options);
    fprintf("Validation accuracy: %f percent.\n", netInfo.FinalValidationAccuracy);
else
    load speechDetectNet
end
EstimatedVADMask = classify(speechDetectNet,featuresValidation.');
EstimatedVADMask = double(EstimatedVADMask);
EstimatedVADMask = EstimatedVADMask.' - 1;