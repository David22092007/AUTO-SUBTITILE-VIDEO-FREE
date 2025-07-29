ffmpeg -i output_video.mp4 -vf "subtitles='translated_subs.srt':force_style='FontName=Segoe UI Black,FontSize=12,PrimaryColour=&H00FFFF&'" -c:v libx264 -preset ultrafast -c:a copy subed_video.mp4
