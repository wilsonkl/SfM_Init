function R=q2R(q)

c2=q(1);

if(abs(abs(c2)-1)>1e-12)

    s2=norm(q(2:4));
    s=2*s2*c2;
    c=2*c2*c2-1;
    n=q(2:4)/s2;

    cc=1-c;

    n1=n(1);                n2=n(2);                n3=n(3);
    n12cc=n1*n2*cc;         n23cc=n2*n3*cc;         n31cc=n3*n1*cc;
    n1s=n1*s;               n2s=n2*s;               n3s=n3*s;

    R(1,1)=c+n1*n1*cc;      R(1,2)=n12cc-n3s;       R(1,3)=n31cc+n2s;
    R(2,1)=n12cc+n3s;       R(2,2)=c+n2*n2*cc;      R(2,3)=n23cc-n1s;
    R(3,1)=n31cc-n2s;       R(3,2)=n23cc+n1s;       R(3,3)=c+n3*n3*cc;
else
    R=eye(3);
end
end