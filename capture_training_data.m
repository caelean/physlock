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
disp('Video capture over');
delete(vid);
clear vid;


vid=VideoReader('./physlock/video/training.avi');
numFrames = vid.NumberOfFrames;
n=numFrames;
for i = 1:1:n
    frame = read(vid,i);
    frame = imresize(frame, 0.125);
    imwrite(frame,['./physlock/images/image' int2str(i), '.jpg']);
    im(i)=image(frame);
end
delete(vid);
clear vid;
