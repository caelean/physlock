function data = capture_data()

% vid = videoinput ('macvideo',2);
% frame = getsnapshot(vid);
% delete(vid);
% clear vid;

frame = imread('test.jpg');
frame = rgb2gray(frame); % converts image to grayscale
level = graythresh(frame); % calculates graythresh level
frame = im2bw(frame,level); % converts image to black and white
frame = imcomplement(frame); % complements image
frame = imclose(frame,strel('disk',6));
[labels,numlabels]=bwlabel(frame); % creates labels
stats = regionprops(labels, 'all'); % calculate statistics

if length(stats) > 1
    frame = imclose(frame,strel('disk',4));
    [labels,numlabels]=bwlabel(frame); % creates labels
    stats = regionprops(labels, 'all'); % calculate statistics
end

f = zeros(length(stats)); % This initializes the f vector
elm = stats(1);
ff = 4*pi*elm.Area/((elm.Perimeter)^2); % This calculates the form factor
data = [ff elm.Area elm.Perimeter];