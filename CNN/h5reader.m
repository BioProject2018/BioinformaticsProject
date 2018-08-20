clear 
close all
clc

disp('Select a .h5 file to open it')
[file,path,~]=uigetfile('*.h5');
name=[path file];
info=hdf5info(name);

datasets={info.GroupHierarchy.Datasets.Name};
data=struct([]);
for i=1:length(datasets)
    data(i).name=datasets{i};
    data(i).values=hdf5read(name,datasets{i});
    if strcmp(datasets{i},'/img_id')==1
        img_id=cell(size(data(i).values));
        for j=1:length(data(i).values)
            img_id{j}=data(i).values(j).Data;
        end
        data(i).values=img_id;
    end
end
clear datasets file indx info name path i j img_id
clc
disp('Data extracted: open data struct to view the results')
    