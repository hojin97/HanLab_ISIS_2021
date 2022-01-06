function [nomalData, normalizer] = retNormalizedDataEachCh(Data,numCh) 

buf = Data;

[val, colids] = max(buf,[],2);
for i=1:numCh
   buf(i,:) = buf(i,:)./val(i);    
%     buf(i,:) 
end
nomalData = buf;
normalizer = val;
end