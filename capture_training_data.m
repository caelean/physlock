function capture_training_data(num_images)

vid = videoinput ('macvideo',2); % declare video link
vid.framesPerTrigger = num_images; % number of images to capture
set(vid,'LoggingMode','disk');
avi = VideoWriter('./physlock/video/training.avi'); % file to write to
set(vid,'DiskLogger',avi);
disp('Video capture starting');
start(vid); % start video capture
wait(vid,Inf);
avi = get(vid,'DiskLogger');
close(avi); % end video capture
delete(vid); % clear variables
clear vid;
disp('Video capture over');

vid=VideoReader('./physlock/video/training.avi'); % read in video
numFrames = vid.NumberOfFrames;
n=numFrames;
disp('Writing data');
data = [];
for i = 1:1:n % iterate through each frame and process it
    frame = read(vid,i);
    frame = rgb2gray(frame); % converts image to grayscale
    level = graythresh(frame); % calculates graythresh level
    frame = im2bw(frame,level); % converts image to black and white
    frame = imcomplement(frame); % complements image
    frame = imclose(frame,strel('disk',6));
    [labels,numlabels]=bwlabel(frame); % creates labels
    stats = regionprops(labels, 'all'); % calculate statistics
    if i % 30 == 0
        imshow(frame); % show every 30th frame to user
    end
    if length(stats) > 1 % too many objects in the image
        frame = imclose(frame,strel('disk',4));
        [labels,numlabels]=bwlabel(frame); % creates labels
        stats = regionprops(labels, 'all'); % calculate statistics
        if length(stats) > 1 % still too many objects, skip image
            disp("invalid number: " + length(stats));
            continue;
        end
    end
    f = zeros(length(stats)); % This initializes the f vector
    elm = stats(1);
    ff = 4*pi*elm.Area/((elm.Perimeter)^2); % This calculates the form factor
    data = [data; (ff) elm.Area elm.Perimeter]; % package data
    im(i)=image(frame);
    
end
dlmwrite("data.csv",data,'-append','precision','%.3f'); % write data to csv
disp(data);
delete(vid); % clear variables
clear vid;
disp('Finished');