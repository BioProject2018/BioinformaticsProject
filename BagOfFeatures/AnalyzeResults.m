clear
close all
clc

R=struct([]);
f=dir;
f={f.name};
for i=1:length(f)
    t=strsplit(f{i},' ');
    if strcmp(t{1},'Results')==0
        f{i}='';
    end
end
f=f(strcmp(f,'')==0);

k=1;
for i=1:length(f)
    f0=dir(f{i});
    f0={f0.name};
    for j=1:length(f0)
        t=strsplit(f0{j},'.');
        if strcmp(t{end},'mat')==1
            name=[f{i} filesep f0{j}];
            load(name)
            R(k).results=Results;
            vs=strsplit(f{i},'_');
            R(k).VocabularySize=str2double(vs{end});
            R(k).name=f0{j};
            k=k+1;
        end
    end
end

for k=1:length(R)
    tmp=R(k).results.onTest.confMat*size(R(k).results.onTest.score,1)/...
        size(R(k).results.onTest.score,2);
    R(k).accuracy_onTest=sum(diag(tmp))/sum(sum(tmp));
    for i=1:length(R(k).results.Categories)
        R(k).(matlab.lang.makeValidName(['pcc' R(k).results.Categories{1,i}]))=tmp(i,i)/sum(tmp(i,:));
    end
    R(k).crossValidation_10fold=R(k).results.onTraining.Score_10fold_cross_validation;
end    
        
R=struct2table(R);
R=R(:,2:end);
writetable(R, 'report.xlsx')