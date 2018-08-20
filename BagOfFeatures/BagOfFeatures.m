clear
close all
clc

plotPrompt={'VocabularySize:','StrongestFeatures:'};
name='BagOfFeatures Parameters';
numlines= [1 50];
%Defaults values to fill the gui
plotDefaultanswer={'200','0.8'};%set default answers inside text fields
options.Resize='on';%make the just open window resizable

%when the user press OK button, the values are recorded from the gui
plotAnswer=inputdlg(plotPrompt,name,numlines,plotDefaultanswer,options);%put in answer the data just inserted from user
VocabularySize=str2double(plotAnswer{1,1});
StrongestFeatures=str2double(plotAnswer{2,1});
currPath = pwd;
addpath([currPath filesep 'functions']);

% 1) Select Training and Test Set and convert them in imageSet variables
[TrainingSets,TestSets,DataSet_name]=prepare_DataSet();

% 2) Extraction of bag of features
bag = bagOfFeatures(TrainingSets,'VocabularySize',VocabularySize,'StrongestFeatures',0.8,'PointSelection','Detector');

% 2.1) Show features vector example:
%   BagOfFeatures object provides an |encode| method for
%   counting the visual word occurrences in an image. It produced an
%   histogram that becomes a new and reduced representation of an image.
mkdir(['Results VocabularySize_' num2str(VocabularySize) filesep 'Examples' filesep DataSet_name filesep...
    'FeaturesVectorHistogram'])
for K=1:ceil(length(TrainingSets(1,1).ImageLocation)/10)
    figure(1)
    figure(2)
    for i=1:length(TrainingSets)
        figure(1)
        set(gcf,'units','points','position',[220,10,500,500])
        subplot(length(TrainingSets),1,i)
        [~,~,name]=featuresVector(bag,TrainingSets(i),K,1);
        figure(2)
        set(gcf,'units','points','position',[10,10,200,500])
        subplot(length(TrainingSets),1,i)
        imshow(read(TrainingSets(i),K))
        title(name)
    end
    x=['Results VocabularySize_' num2str(VocabularySize) filesep,'Examples' filesep, DataSet_name, filesep,...
        'FeaturesVectorHistogram',filesep, num2str(K),'.png'];
    saveas(figure(1),x)
end

% 3) Training a Multi-class SVM classifier using the features extracted
%    with bag of features.

% 3.1) Tuning on Multi-class SVM parameters
Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100,1000];
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10];
Tols=[1e-4,1e-3,1e-2,1e-1];
score=0;
C=Cs(1);
gamma=gammas(1);
Tol=Tols(1);
[X,Y]=createFeaturesMatrix(bag,TrainingSets);
n=0;
for i=1:length(Cs)
    for j=1:length(gammas)
        for k=1:length(Tols)
            n=n+1;
            t = templateSVM('KernelFunction','rbf',...
                'KernelScale',gammas(j),'BoxConstraint',Cs(i),...
                'DeltaGradientTolerance',Tols(k));
            SVMModel = fitcecoc(X,Y,'Learners',t);
            CVSVMModel = crossval(SVMModel);
            %10-fold cross-validation loss
            L = kfoldLoss(CVSVMModel);
            disp(['tuning on parameters (' num2str(n) '/'...
                num2str(length(Cs)*length(gammas)*length(Tols)) ')'])
            if 1-L>score
                score=1-L;
                C=Cs(i);
                gamma=gammas(j);
                Tol=Tols(k);
            end
        end
    end
end
Results=struct([]);
Results(1).onTraining.Score_10fold_cross_validation=score;

% 4) Evaluate Classifier Performance on Test Set
t = templateSVM('KernelFunction','rbf',...
    'KernelScale',gamma,'BoxConstraint',C,'DeltaGradientTolerance',Tol);
categoryClassifier = trainImageCategoryClassifier(TrainingSets,...
    bag,'LearnerOptions',t);

disp('Evaluating Classifier Performance on Test Set')
[Results(1).onTest.confMat,Results(1).onTest.knownLabelIdx,...
    Results(1).onTest.predictedLabelIdx,Results(1).onTest.score]...
    = evaluate(categoryClassifier, TestSets);
Results(1).onTraining.ImageLocation=[];
Results(1).onTest.ImageLocation=[];
Results(1).Categories=cell(1,length(TrainingSets));
for i=1:length(TrainingSets)
    Results(1).onTraining.ImageLocation=[Results(1).onTraining.ImageLocation; TrainingSets(1,i).ImageLocation'];
    Results(1).onTest.ImageLocation=[Results(1).onTest.ImageLocation; TestSets(1,i).ImageLocation'];
    Results(1).Categories{1,i}=TrainingSets(1, i).Description;
end

% 5) Examples of visualization
mkdir(['Results VocabularySize_' num2str(VocabularySize) filesep 'Examples' filesep DataSet_name filesep...
    'WordsOnImage'])
% Get the strongest Wn words for each category
Wstrong_n=getStrongestWords(bag,TrainingSets,ceil(VocabularySize/5),0);
figure(3)
for K=1:ceil(length(TestSets(1,1).ImageLocation)/5)
    for i=1:length(TestSets)
        [~,words,name]=featuresVector(bag,TestSets(i),K,0);
        img=read(TestSets(i),K);
        labelIdx=Results(1).onTest.predictedLabelIdx((i-1)*length(...
            TestSets(i).ImageLocation)+K);
        cat=categoryClassifier.Labels{labelIdx};
        subplot(1,length(TestSets),i)
        imshow(img)
        hold on
        % plot all fatures detected
        plot(words.Location(:,1),words.Location(:,2),'ro')
        % strongest words of detected category
        strong4ThisCathegory=Wstrong_n(:,labelIdx);
        % detected word
        detected_words=words.WordIndex;
        % check if detected word is in strong word list
        if length(detected_words)>1
            ind=zeros(1,length(detected_words));
            for j=1:length(detected_words)
                ind(j)=max(detected_words(j)==strong4ThisCathegory);
            end
            % plot strong features
            plot(words.Location(logical(ind),1),words.Location(logical(ind),2),'g*')
        end
        text(20,40,cat,'Color','yellow','FontSize',30)
        title(name)
    end
    
    x=['Results VocabularySize_', num2str(VocabularySize), filesep,'Examples' filesep, DataSet_name, filesep,...
        'WordsOnImage',filesep, num2str(K),'.png'];
    saveas(gcf,x)
end


Results(1).onTest.knownLabelIdx=Results(1).Categories(Results(1).onTest.knownLabelIdx)';
Results(1).onTest.predictedLabelIdx=Results(1).Categories(Results(1).onTest.predictedLabelIdx)';
% Save Results
x=['Results VocabularySize_', num2str(VocabularySize), filesep,'Results-', DataSet_name, '.mat'];
save(x,'Results')

