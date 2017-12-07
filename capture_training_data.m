function capture_training_data()

vid = videoinput ('macvideo',2);
vid.framesPerTrigger = 100;
set(vid,'LoggingMode','disk');
avi = VideoWriter('./video/training.avi');
set(vid,'DiskLogger',avi);
start(vid);
wait(vid,Inf); 
avi = get(vid,'DiskLogger');
close(avi);
delete(vid);
clear vid;


vid=VideoReader('./video/training.avi');
numFrames = vid.NumberOfFrames;
n=numFrames;
for i = 1:2:n
    frame = read(vid,i);
    frame = imresize(frame, 0.125);
    imwrite(frame,['./images/Image' int2str(i), '.jpg']);
    im(i)=image(frame);
end
delete(vid);
clear vid;
