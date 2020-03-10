
%%
clc
category = {'PAUL','VINCENT','OLIVER'};
rootfolder = fullfile('images folder');
imds =  imageDatastore(fullfile(rootfolder,category),'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%%
tbl = countEachLabel(imds);
minCount = min(tbl{:,2});
imds = splitEachLabel(imds,minCount,'randomized');
%%
image1 = readimage(imds,27);
image1 = imresize(image1,[300 300]);
[hog,vis] =extractHOGFeatures(image1,'CellSize',[2 2]); 
imshow(image1); hold on; plot(vis);
[training,test]=splitEachLabel(imds,0.8);
%%
numImages = numel(training.Files);

trainingFeatures= zeros(numImages,length(hog),'single');

for j = 1:numImages
    I = readimage(training,j);
    I = imresize(I,[300 300]);
    trainingFeatures(j,:) = extractHOGFeatures(I,'CellSize',[2 2]);
    
end

trainingLabels = training.Labels;

%%
classifier = fitcecoc(trainingFeatures,trainingLabels);
YTest = test.Labels;
%%
unkn=readimage(test,14);
%imshow(unkn)
%unkn = imread('IMG_20200307_115537.jpg');
%imshow(unknown)
unkn=imresize(unkn,[300 300]);

unknfeats=extractHOGFeatures(unkn,'CellSize',[2 2]);
functlabelr= predict(classifier,unknfeats);

imshow(unkn); hold on; title(functlabelr)



%%


