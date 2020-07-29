# Check if FFMPEG is installed
YTDL=youtube-dl



mkdir -p videos

		# # echo "$entry"
		# if [ "$1" -eq 0 ]; then
		# 	while read line
		# 		do
		# 			# Use -f 22/best to download in whatever best format and quality available if not 22 i.e. mp4 720p
		# 			$YTDL --match-filter 'duration<=3600' -f 'bestvideo[width=1280][fps=30]'  "http://www.youtube.com/watch?v=$line" -o ./videos/"%(title)s-%(id)s.%(ext)s"
		# 		done < Apartment
		# else
limit=$1
rm -rf log3.txt
while read line
	do
		if [ "$limit" -gt 0 ]; then
			limit=$(($limit-1))
			$YTDL --match-filter 'duration<=3600' -f 'bestvideo[width=1280][fps=30]' "http://www.youtube.com/watch?v=$line" -o ./videos/"%(title)s-%(id)s.%(ext)s" 2>&1 | tee log3.txt
			error=$(grep -c ERROR log3.txt)
			if [ "$error" -gt 0 ]; then
				limit=$(($limit+1))
				# echo $line >> outstandingVids1.txt
			fi
			echo $limit
			rm -rf log3.txt
		else
			break
		fi
	done < "$2"
		# fi
