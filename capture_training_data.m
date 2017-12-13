function capture_training_data(num_images)

vid = videoinput ('macvideo',2);
vid.framesPerTrigger = num_images;
set(vid,'LoggingMode','disk');
avi = VideoWriter('./physlock/video/training.avi');
set(vid,'DiskLogger',avi);
disp('Video capture starting');
start(vid);
wait(vid,Inf);
avi = get(vid,'DiskLogger');
close(avi);
delete(vid);
clear vid;
disp('Video capture over');

vid=VideoReader('./physlock/video/training.avi');
numFrames = vid.NumberOfFrames;
n=numFrames;
disp('Writing data');
data = [];
for i = 1:1:n
    frame = read(vid,i);
    frame = rgb2gray(frame); % converts image to grayscale
    level = graythresh(frame); % calculates graythresh level
    frame = im2bw(frame,level); % converts image to black and white
    frame = imcomplement(frame); % complements image
    frame = imclose(frame,strel('disk',6));
    [labels,numlabels]=bwlabel(frame); % creates labels
    stats = regionprops(labels, 'all'); % calculate statistics
    if i % 30 == 0
        imshow(frame);
    end
    if length(stats) > 1
        frame = imclose(frame,strel('disk',4));
        [labels,numlabels]=bwlabel(frame); % creates labels
        stats = regionprops(labels, 'all'); % calculate statistics
        if length(stats) > 1
            disp("invalid number: " + length(stats));
            continue;
        end
    end
    f = zeros(length(stats)); % This initializes the f vector
    elm = stats(1);
    ff = 4*pi*elm.Area/((elm.Perimeter)^2); % This calculates the form factor
    data = [data; (ff) elm.Area elm.Perimeter];
    im(i)=image(frame);
    
end
dlmwrite("data.csv",data,'-append','precision','%.3f');
disp(data);
delete(vid);
clear vid;
disp('Finished');