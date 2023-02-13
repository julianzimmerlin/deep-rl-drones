
set FRAME_RATE=24
set RESOLUTION=640x480

ffmpeg -r %FRAME_RATE% -f image2 -s %RESOLUTION% -i frame_%%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p video.mp4