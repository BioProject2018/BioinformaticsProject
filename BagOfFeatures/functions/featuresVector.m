function [featureVector,words,name]=featuresVector(bag,imageSet,n,verbose)

% Inputs:
% - bag is a bagOfFeatures object
% - imageSet is an imageSet object
% - n is an integer. It is the index of an image in a imageSet
% Outputs:
% - featureVector is an array. It is an histogram about the 
%   visual word occurrences in an image.
% - name is a string. It is the name of the selected image

% BagOfFeatures object provides an |encode| method for
% counting the visual word occurrences in an image. It produced a histogram
% that becomes a new and reduced representation of an image.
img = read(imageSet, n);
[featureVector,words] = encode(bag, img);

% get name of the image
path=imageSet.ImageLocation{1,n};
path=strsplit(path,filesep);
name=path{1,end};
name=strsplit(name,'.');
name=name{1,1};
name(name=='_')='-';

if verbose==1
    % Plot the histogram of visual word occurrences
    bar(featureVector)
    title([name ': Visual word occurrences'])
    xlabel('Visual word index')
    ylabel('Frequency of occurrence')
end

end