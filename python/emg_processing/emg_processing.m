%----------------------------------------------------------\
% Myodata_All_Processing.m
% Author : Han U. Yoon (2021/01/26)
% Purpose: To give some nice exprience to my commrade J.H. and H.J.
% Revise note: 
%----------------------------------------------------------\

function emg_processing(instrExercise,innumTrial,strcmap,strAAFTnum)

cutoff_freq = 7.5;
fs = 50;
ts = 0.020; % sampling time
windowSize = 10;

global filteredResult rmsResult avgResult datalength 

%%-- read file
% strExercise = 'biceps';
% strExercise = 'triceps';
% numTrial = 1;
strExercise = instrExercise;
numTrial = innumTrial;
% filename = './data/biceps/biceps_emg_20ms_trial (11).txt';
% filename = './data/triceps/triceps_emg_20ms_trial (41).txt';

% % filename = sprintf('./data/%s/%s_emg_20ms_trial (%d).txt',strExercise, strExercise, numTrial);
% filename = sprintf('./data_AAFT(20)/%s/%s_emg_20ms_trial (%d).txt',strExercise, strExercise, numTrial);

% switch strAAFTnum
%     case 'AAFT0'
%         filename = sprintf('./data/%s/%s_emg_20ms_trial (%d).txt',strExercise, strExercise, numTrial);
%     case 'AAFT5'
%         filename = sprintf('./data_AAFT(5)/%s/%s_emg_20ms_trial (%d).txt',strExercise, strExercise, numTrial);
%     case 'AAFT10'
%         filename = sprintf('./data_AAFT(10)/%s/%s_emg_20ms_trial (%d).txt',strExercise, strExercise, numTrial);
%     case 'AAFT20'
%         filename = sprintf('./data_AAFT(20)/%s/%s_emg_20ms_trial (%d).txt',strExercise, strExercise, numTrial);
%     otherwise
%         error('Wrong AAFT options');
% end

switch strAAFTnum
    case {'AAFT(0)','AAFT(2)','AAFT(3)','AAFT(4)','AAFT(5)','AAFT(10)','AAFT(20)'}
        filename = sprintf('./emg_raw/%s/%s/%s_emg_20ms_trial (%d).txt',strAAFTnum,strExercise, strExercise, numTrial)
    otherwise
        error('Wrong AAFT options');
end

fid = fopen(filename,'r');

N = 80; % need to be elaborated according to data length

datafieldspec = '%f   %f %f %f %f  %f %f %f %f  %f';
S_data = textscan(fid,datafieldspec);
fclose(fid);
%---------------------------------------------------------------------------\
% S_data{1,1} = time stamp, sampling rate ts = 20/1000 sec = 20ms
% S_data{1,2} -- S_data{1,9} = EMG ch1 -- EMG ch 8 value
% S_data{1,10}  = elbow angle
%---------------------------------------------------------------------------\
% Indexing example
% S{1,1}(1): The first time stamp = 1
% S{1,3}(10): EMG ch2 data at time 10x20ms [sec]
%---------------------------------------------------------------------------\

datalength = length(S_data{1,1});
% datalength = N;

numEMG = 8;
LPfiltered_EMG = zeros(numEMG,datalength);
movingRMS_EMG = zeros(numEMG,datalength);

for i=1:numEMG
    
    Voltage = ( S_data{1,2+i-1}-mean(S_data{1,2+i-1}(1:(round(datalength*0.1)))) );
%     Voltage = S_data{1,2+i-1};
    absVoltage = abs(Voltage);
    
    %% -- original data
    %figure(1);
    plot(Voltage);
    % -- figure setting (safe to ignore this!)
    xAxis = ts*get(gca,'XTick');
    set(gca,'XTickLabel',xAxis);
%     set(gca,'YLim',[-1.5 1.5]);    
    strtitle = sprintf('original EMG%d',i);
    title(strtitle);
    xlabel('[sec]');
    ylabel('[v]')

    %% - processed data
    %figure(2);
    % -- apply butterworth filter
%     cutoff_freq = 15;
%     fs = 50;
    butterWorthLP(absVoltage,cutoff_freq,fs); 
    hf = plot(filteredResult,'g'); hold on;
    LPfiltered_EMG(i,:) = filteredResult;
    
    % -- moving RMS
%     windowSize = 10;
    movingRMS(absVoltage,windowSize);
    hr = plot(rmsResult,'r'); hold off;
    movingRMS_EMG(i,:) = rmsResult;
    
    % -- figure setting (safe to ignore this!)
    xAxis = ts*get(gca,'XTick');
    set(gca,'XTickLabel',xAxis);
%     set(gca,'YLim',[-0.1 1.0]);
    set(gca,'YLim',[0 150]);
    strtitle = sprintf('processed EMG%d',i);
    title(strtitle);
    xlabel('[sec]');
    ylabel('[v]');
    legend([hf hr],'butterworth LP-filtered','moving RMS','Location','SouthWest');
    
%     pause;
    
end

%% - Process Joint Angle

numJANGLE = 1;
LPfiltered_JANGLE = zeros(numJANGLE,datalength);
movingRMS_JANGLE = zeros(numJANGLE,datalength);
movingAVG_JANGLE = zeros(numJANGLE,datalength);

for i=1:numJANGLE
    % S_data{1,10}: elbow angle
%     Degree = ( S_data{1,2+numEMG+i-1}-mean(S_data{1,2+numEMG+i-1}(1:(round(datalength*0.1)))) );
%     absVoltage = abs(Voltage);
    
%     Degree = S_data{1,2+numEMG+i-1};
    Degree = abs(S_data{1,2+numEMG+i-1}); % or need to be rearranged from 0 to 2*pi
    
    %% -- original data
    %figure(1);
    plot(Degree);
    % -- figure setting (safe to ignore this!)
    xAxis = ts*get(gca,'XTick');
    set(gca,'XTickLabel',xAxis);
%     set(gca,'YLim',[-1.5 1.5]);    
    strtitle = sprintf('original JOINT ANGLE%d',i);
    title(strtitle);
    xlabel('[sec]');
    ylabel('[deg]')

    %% - processed data
    %figure(2);
    % -- apply butterworth filter
%     cutoff_freq = 15;
%     fs = 50;
    butterWorthLP(Degree,cutoff_freq,fs); 
    hf = plot(filteredResult,'g'); hold on;
    LPfiltered_JANGLE(i,:) = filteredResult;
    
    % -- moving RMS
%     windowSize = 10;
    movingRMS(Degree,windowSize);
%     hr = plot(rmsResult,'r'); hold off;
    movingRMS_JANGLE(i,:) = rmsResult;
    
    movingAVG(Degree,windowSize);
    hr = plot(avgResult,'r'); hold off;
    movingAVG_JANGLE(i,:) = avgResult;
    
    % -- figure setting (safe to ignore this!)
    xAxis = ts*get(gca,'XTick');
    set(gca,'XTickLabel',xAxis);
%     set(gca,'YLim',[-0.1 1.0]);
    set(gca,'YLim',[0 180]);
    strtitle = sprintf('processed JOINT ANGLE%d',i);
    title(strtitle);
    xlabel('[sec]');
    ylabel('[deg]');
%     legend([hf hr],'butterworth LP-filtered','moving RMS','SouthEast');
    legend([hf hr],'butterworth LP-filtered','moving AVG','Location','SouthWest');
    
%     pause;
    
end

matfilename = 'filteredData.mat';
save(matfilename,'LPfiltered_EMG','movingRMS_EMG','numEMG','LPfiltered_JANGLE','movingRMS_JANGLE','movingAVG_JANGLE','numJANGLE','datalength');

IntgetralWindowSize = 5; 
k0 = 1;
kf = 80;

% if datalenth < 80 then fedding avg of last 5 values upto 80
if datalength < 80
    
    LPfiltered_EMG = [ LPfiltered_EMG, zeros(numEMG,(kf-datalength)) ];
    LPfiltered_JANGLE = [ LPfiltered_JANGLE, zeros(numJANGLE,(kf-datalength)) ];
    movingRMS_EMG = [ movingRMS_EMG, zeros(numEMG,(kf-datalength)) ];    
    movingRMS_JANGLE = [ movingRMS_JANGLE, zeros(numJANGLE,(kf-datalength)) ];
    
    for k=(datalength+1):kf
%         size(LPfiltered_EMG(:,k))
%         size(mean(LPfiltered_EMG(:,datalength-5:datalength),2))
        LPfiltered_EMG(:,k) = mean(LPfiltered_EMG(:,datalength-5:datalength),2);
        LPfiltered_JANGLE(:,k) = mean(LPfiltered_JANGLE(:,datalength-5:datalength),2);
        movingRMS_EMG(:,k) = mean(movingRMS_EMG(:,datalength-5:datalength),2);
        movingRMS_JANGLE(:,k) = mean(movingRMS_JANGLE(:,datalength-5:datalength),2);
    end
end
%--------------------------------------------------------

sigDATA1 = [
            movingRMS_EMG(:,k0:kf)           
        ];
sigDATA2 = [
             movingRMS_JANGLE(:,k0:kf)
        ];
%--------------------------------------------------------
%     sigDATA1 = [
%             LPfiltered_EMG(:,k0:kf)           
%         ];
% sigDATA2 = [
%              LPfiltered_JANGLE(:,k0:kf)
%         ];
%--------------------------------------------------------

sigDATAmod1 = sigDATAmodification(sigDATA1,IntgetralWindowSize,'integrate');
sigDATAmod2 = sigDATAmodification(sigDATA2,IntgetralWindowSize,'integrate');
% sigDATAmod1 = sigDATAmodification(sigDATA1,IntgetralWindowSize,'average');
% sigDATAmod2 = sigDATAmodification(sigDATA2,IntgetralWindowSize,'average');

%- Normalize sigDATA
sigDATAnormal1 = retNormalizedData(sigDATAmod1,numEMG);  
sigDATAnormal2 = retNormalizedDataEachCh(sigDATAmod2,numJANGLE);  

sigDATAnormal = [
            round(sigDATAnormal1, 3);
            round(sigDATAnormal2, 3)
        ];

output_path = "./emg_result/"+ strAAFTnum + "/" + strExercise + "/" + numTrial + ".txt";
fp = fopen(output_path, 'w');
writematrix(sigDATAnormal, output_path, 'Delimiter', 'tab');
fclose(fp);
%pause()
end

%------------------------- subfunctions --------------------------%

function butterWorthLP(data,cutoff_freq,sampling_freq)
global filteredResult

[B,A]=butter(4,cutoff_freq/(sampling_freq/2),'low');
% absdata = abs(data);
filteredResult = filtfilt(B,A,data);

end

function movingRMS(data, windowSize)
global rmsResult datalength

hWD = windowSize/2; % 10

for i=1:datalength
    
    if i < hWD
        x = data(1:i+hWD);
        y(i) = norm(x)/sqrt(length(x));
    else if i > datalength-hWD
             x = data(i-hWD:datalength);
             y(i) = norm(x)/sqrt(length(x));
        else % normal range
             x = data(i- hWD+1: i+hWD);
             y(i) = norm(x)/sqrt(length(x));   
         end
    end
    
rmsResult = y;    
    
end

end

function movingAVG(data, windowSize)
global avgResult datalength

hWD = windowSize/2; % 10

for i=1:datalength
    
    if i < hWD
        x = data(1:i+hWD);
%         y(i) = norm(x)/sqrt(length(x));
        y(i) = mean(x);
    else if i > datalength-hWD
             x = data(i-hWD:datalength);
%              y(i) = norm(x)/sqrt(length(x));
             y(i) = mean(x);
        else % normal range
             x = data(i- hWD+1: i+hWD);
%              y(i) = norm(x)/sqrt(length(x));   
             y(i) = mean(x);  
         end
    end
    
avgResult = y;    
    
end

end
