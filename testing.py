from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


video_id = "oX7OduG1YmI"
ytt_api = YouTubeTranscriptApi()
transcript_list = ytt_api.list(video_id)

print(transcript_list)

# transcript = transcript_list.find_transcript(['en'])
# translated_transcript = transcript.translate('de')
# print(translated_transcript.fetch())