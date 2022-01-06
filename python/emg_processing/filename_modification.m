clear all;
close all;

% Hammer, Rvcurl, Bi, Tri
folder_name = './emg_raw/AAFT(2)/Hammer/';
list = dir(folder_name);

% 62   122   182    242   302   602   1202
for i = 3:122
   c_name = list(i).name;
   n_name = join(['Hammer_emg_20ms_trial (', num2str(i-2) ,').txt']);
   movefile([folder_name c_name], [folder_name n_name]);
end
