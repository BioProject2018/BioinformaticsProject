function [X,y]=createFeaturesMatrix(bag,imageSet)

N=size(imageSet,2);
k=0;
for i=1:N
    for j=1:length(imageSet(1, i).ImageLocation)
        k=k+1;
    end
end
X=zeros(k,bag.VocabularySize);
y=zeros(k,1);

k=1;
for i=1:N
    for j=1:length(imageSet(1, i).ImageLocation)
        y(k)=i;
        img = read(imageSet(1,i), j);
        X(k,:)=encode(bag, img);
        k=k+1;
    end
end