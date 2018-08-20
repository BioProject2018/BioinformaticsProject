clear
close all
clc

folders=dir;
folders={folders.name};
% Download TIA Lab - Stain Normalization Toolbox if it is not present
if sum(strcmp(folders,'stain_normalisation_toolbox'))==0
    url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/software/sntoolbox/stain_normalisation_toolbox_v2_2.zip';
    websave('sntoolbox.zip',url);
    unzip('sntoolbox.zip');
    delete 'sntoolbox.zip'
end

currentFolder = pwd;
addpath([currentFolder filesep 'stain_normalisation_toolbox']);
cd stain_normalisation_toolbox
install
cd ..

% Define Training and Test Set folders
disp('Select Training Set Folder')
TrainFolder=uigetdir(currentFolder,'Select Training Set Folder');
clc
disp('Select Test Set Folder')
TestFolder=uigetdir(currentFolder,'Select Test Set Folder');
clc

% Create ImageDatastore variables for Training and Test Set
TrainDS = imageDatastore(TrainFolder,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
TestDS = imageDatastore(TestFolder,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');

% Create a folder where store new DataSets
mkdir('DataSets');

% Select type of new DataSet to create
list = {'-GrayScale','-ReinhardNormalization','-Hematoxylin'};
[indx,tf] = listdlg('ListString',list);

f = waitbar(0,'','Name','Processing...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');

TargetImage=imread(TrainDS.Files{24,1});
step=0;
steps=length(TrainDS.Files)+length(TestDS.Files);
for r=1:2
    if r==1
        DS=TestDS;
        T='Test';
    else
        DS=TrainDS;
        T='Training';
    end
    C=unique(DS.Labels);
    Data=DS.Files;
    verbose=0;
    for k=1:length(indx)
        choice=char(list(indx(k)));
        path0=['DataSets' filesep 'DataSet' choice filesep T choice filesep];
        for i=1:length(C)
            mkdir([path0 char(C(i,1))])
        end
        
        switch choice
            case '-GrayScale'
                for i=1:length(Data)
                    name=Data{i,1};
                    I=imread(name);
                    % Convert image in gray scale
                    J=rgb2gray(I);
                    I=uint8(zeros(size(I)));
                    I(:,:,1)=J;
                    I(:,:,2)=J;
                    I(:,:,3)=J;
                    name=strsplit(name,filesep);
                    name=strjoin({name{1,end-1:end}},filesep);
                    name0=[path0 name];
                    imwrite(I,name0);
                    step=step+1;
                    % Check for clicked Cancel button
                    if getappdata(f,'canceling')
                        break
                    end
                    % Update waitbar and message
                    waitbar(step/steps,f,sprintf([num2str(step) '/' num2str(steps)]))
                end
            case '-ReinhardNormalization'
                for i=1:length(Data)
                    name=Data{i,1};
                    I=imread(name);
                    % Normalize source image respect to a target image 
                    % using using Reinhard's method 
                    I= Norm( I, TargetImage, 'Reinhard', verbose );
                    name=strsplit(name,filesep);
                    name=strjoin({name{1,end-1:end}},filesep);
                    name0=[path0 name];
                    imwrite(I,name0);
                    step=step+1;
                    % Check for clicked Cancel button
                    if getappdata(f,'canceling')
                        break
                    end
                    % Update waitbar and message
                    waitbar(step/steps,f,sprintf([num2str(step) '/' num2str(steps)]))
                end
            case '-Hematoxylin'
                for i=1:length(Data)
                    name=Data{i,1};
                    I=imread(name);
                    % Estimate the stain separation matrix using the Stain 
                    % Colour Descriptor method
                    SCDMatrix = EstUsingSCD(I);
                    % Deconvolution of image into its constituent stain
                    % channels
                    stains=Deconvolve(I, SCDMatrix, verbose);
                    % Obtain conversion in pseudo-colour stains
                    [H, ~, ~] = PseudoColourStains( stains, SCDMatrix);
                    name=strsplit(name,filesep);
                    name=strjoin({name{1,end-1:end}},filesep);
                    name0=[path0 name];
                    imwrite(H,name0);
                    step=step+1;
                    % Check for clicked Cancel button
                    if getappdata(f,'canceling')
                        break
                    end
                    % Update waitbar and message
                    waitbar(step/steps,f,sprintf([num2str(step) '/' num2str(steps)]))
                end
        end
    end
end
delete(f)