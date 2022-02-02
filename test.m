data=fopen("ch0_k-6_1.bin",'rb');
noisySpeech=fread(data,'int16');
noisySpeech=noisySpeech./32767;
noisySpeech=noisySpeech(1:16000*20);
% [noisySpeech,fs] = audioread('train-clean-100\train-clean-100\4195\186238\4195-186238-0003.flac');
% [noisySpeech,fs] = audioread('google_speech\test\down\0f250098_nohash_0.wav');
% [noisySpeech,fs] = audioread('alex_noisy.wav');
% noisySpeech=resample(noisySpeech,16000,fs);
features = extract(afe,noisySpeech);
features = (features - mean(features,1)) ./ std(features,[],1);
features = features';
decisionsCategorical = classify(speechDetectNet,features);
decisionsWindow = 1.2*(double(decisionsCategorical)-1);
decisionsSample = [repelem(decisionsWindow(1),numel(afe.Window)), ...
                   repelem(decisionsWindow(2:end),numel(afe.Window)-afe.OverlapLength)];
t = (0:numel(decisionsSample)-1)/afe.SampleRate;
plot(t,noisySpeech(1:numel(t)), ...
     t,decisionsSample);
xlabel('Time (s)')
ylabel('Amplitude')
legend('Noisy Speech','VAD','Location','southwest')