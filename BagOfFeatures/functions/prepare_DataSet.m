function [TrainingSets,TestSets,DataSet_name]=prepare_DataSet(minSetCont_Randomize)

if nargin==0
    minSetCont_Randomize=0;
end

% Define Training and Test Set folders
currentFolder = pwd;
disp('Select Training Set Folder')
TrainFolder=uigetdir(currentFolder,'Select Training Set Folder');
clc
disp('Select Test Set Folder')
TestFolder=uigetdir(currentFolder,'Select Test Set Folder');
clc
DataSet_name=strsplit(TestFolder,filesep);
DataSet_name=DataSet_name{end-1};

% Get subfolders
sub_training=dir(TrainFolder);
sub_training={sub_training.name};
sub_training=sub_training(1,3:end);
sub_test=dir(TestFolder);
sub_test={sub_test.name};
sub_test=sub_test(1,3:end);
assert(isempty(setdiff(sub_training,sub_test)),...
    'To obtain the accuracy of classification Training and Test subfolders (classes) must be equal')

% Make Training and Test set imageSet variables
TrainingSets=[];
TestSets=[];
for i=1:length(sub_training)
    TrainingSets =[TrainingSets imageSet(fullfile(TrainFolder,sub_training{1,i}))];
    TestSets =[TestSets imageSet(fullfile(TestFolder,sub_training{1,i}))];
end

if minSetCont_Randomize==1
    % Determine the smallest amount of images in a category for Training Set
    minSetCount = min([TrainingSets.Count]);
    % Use partition method to trim the set.
    TrainingSets = partition(TrainingSets, minSetCount, 'randomize');
    % Determine the smallest amount of images in a category for Test Set
    minSetCount = min([TestSets.Count]);
    %Use partition method to trim the set.
    TestSets = partition(TestSets, minSetCount, 'randomize');
end

% Display number of images for each category in Training and Test Set
disp('Number of images for each category in Training and Test Set:')
Category={TrainingSets.Description}';
Training=[TrainingSets.Count]';
Test=[TestSets.Count]';
T = table(Category,Training,Test);
disp(T)
end