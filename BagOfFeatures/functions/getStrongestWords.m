function Wstrong_n=getStrongestWords(bag,imageSet,Wn,verbose)
% Method to identify strongest Wn words for each category

if nargin==3
    verbose=0;
end

assert(Wn<=bag.VocabularySize,...
    'number of strongest words must be minor or equal than vocabulary size')

% Get histogram of the visual word occurrences for each image and store
% it in the proper matrix in a struct
fm=struct([]);
N = length(imageSet);
for i=1:N
    [fm(i).featureVectors,fm(i).words]= encode(bag,imageSet(i));
end

Wstrong_n=zeros(Wn,N);

for j=1:N
    loc=fm(j).words;
    wordVect=[];
    for i = 1:length(loc)
        wordVect=[wordVect;loc(i).WordIndex];
    end
    
    WordHistogram=zeros(bag.VocabularySize,1);
    for w=1:bag.VocabularySize
        WordHistogram(w)=sum(wordVect==w)/length(wordVect);
    end
    [~,ind]=sort(WordHistogram,'descend');
    f100 = ind(1:Wn);
    
    if verbose==1
        plot(1:bag.VocabularySize,WordHistogram,'r*')
        hold on
        plot(f100,WordHistogram(ind(1:Wn)),'g*')
    end
    
    Wstrong_n(:,j)=f100;
end