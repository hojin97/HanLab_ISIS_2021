function [nomalData, normalizer] = retNormalizedData(Data,numCh) 

buf = Data;

[val, colids] = max(buf,[],2);
for i=1:numCh
   temp(i) = buf(i,colids(i));    
end

[val, rowidx] = max(temp);   
colidx = colids(rowidx);
normalizer = buf(rowidx,colidx);
nomalData = buf ./ normalizer;

end