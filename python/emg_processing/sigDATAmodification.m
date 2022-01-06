function sigDATAGmod = sigDATAmodification(sigDATA,WindowSize,modoption)

switch modoption
    case 'average'
        tempBuf = averageDATA(sigDATA,WindowSize);
        %
    case 'integrate'
        tempBuf = integrateDATA(sigDATA,WindowSize);
        %
    otherwise
        disp('Wrong option number... Data will be averaged');
        tempBuf = averageDATA(sigDATA,WindowSize);
        %average
end

sigDATAGmod = tempBuf;

end

%---------------- subfunctions -----------------
function Buf = averageDATA(sigDATA,WindowSize)
n = length(sigDATA(1,:));
K = floor(n/WindowSize);

for i=1:K
    Buf(:,i) = abs(mean(sigDATA(:,WindowSize*(i-1)+1:WindowSize*i),2));  
end
    
end


function Buf = integrateDATA(sigDATA,WindowSize)
        
n = length(sigDATA(1,:));
K = floor(n/WindowSize);

for i=1:K
    Buf(:,i) = abs(sum(sigDATA(:,WindowSize*(i-1)+1:WindowSize*i),2));    
end
    
end

%-------------------------------------------------------