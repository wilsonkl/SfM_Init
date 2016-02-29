function [E,e]=CompareRotations(R1,R2)

e=zeros(size(R1,3),1);
for i=1:size(R1,3)
    if(sum(sum((R1(:,:,i)==0)))==9||any(any(isnan(R1(:,:,i))))||...
        sum(sum((R2(:,:,i)==0)))==9||any(any(isnan(R2(:,:,i)))));
        e(i,1)=NaN;
    else
        e(i,1)=acos(max(min((R1(1,:,i)*R2(1,:,i)'+R1(2,:,i)*R2(2,:,i)'+R1(3,:,i)*R2(3,:,i)'-1)/2,1),-1));
    end
end
e=e*180/pi;
i=~isnan(e);
E=[mean(e(i)) median(e(i)) sqrt(e(i)'*e(i)/sum(i))];
disp(strcat('Mean    Angular Error (in Degree) : ',num2str(E(1,1))));
disp(strcat('Median Angular Error (in Degree) : ',num2str(E(1,2))));
disp(strcat('RMS     Angular Error (in Degree) : ',num2str(E(1,3))));
hist(e,180);

end
