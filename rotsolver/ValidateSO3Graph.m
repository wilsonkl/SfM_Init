function [E,e]=ValidateSO3Graph(R,RR,I)
RRa=zeros(size(RR));
for i=1:size(I,2)
    RRa(:,:,i)=R(:,:,I(2,i))*R(:,:,I(1,i))';
end
[E,e]=CompareRotations(RRa,RR);
end