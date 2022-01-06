clear all;
close all;
addpath('./export_fig-master/export_fig-master');  

AAFTList = {'AAFT(0)', 'AAFT(2)','AAFT(3)','AAFT(4)','AAFT(5)','AAFT(10)','AAFT(20)'};
%,'AAFT(2)','AAFT(3)','AAFT(4)','AAFT(5)','AAFT(10)','AAFT(20)'
ColormapList = {'jet','bone'};
ExerciseList = {'Bi', 'Tri', 'Rvcurl', 'Hammer'};


for p=1:2    
    strAAFTnum = AAFTList{p};    
    for q=1:1
        strcmap = ColormapList{q};            
        for r=1:4
        strExercise = ExerciseList{r};    
        
            switch strAAFTnum
                case 'AAFT(0)'
                    K = 60;
                case 'AAFT(2)'
                    K = 120;
                case 'AAFT(3)'
                    K = 180;
                case 'AAFT(4)'
                    K = 240;        
                case 'AAFT(5)'
                    K = 300;        
                case 'AAFT(10)'
                    K = 600;
                case 'AAFT(20)'
                    K = 1200;
            end
            
%             for i=1:2
            for i=1:K
               %emg_processing(strExercise,i,strcmap,strAAFTnum); 
               emg_processing_midAng(strExercise,i,strcmap,strAAFTnum); 
            end        
        end
    end
end