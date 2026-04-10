import { useState } from 'react'

interface VideoPlayerProps {
  videoId: string
  hasAnnotatedVideo: boolean
}

type VideoMode = 'annotated' | 'original'

function VideoPlayer({ videoId, hasAnnotatedVideo }: VideoPlayerProps) {
  const [mode, setMode] = useState<VideoMode>(
    hasAnnotatedVideo ? 'annotated' : 'original'
  )

  const videoUrl =
    mode === 'annotated'
      ? `/api/videos/${videoId}/stream/annotated`
      : `/api/videos/${videoId}/stream`

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Video Playback</h2>
        {hasAnnotatedVideo && (
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            <button
              className={`px-3 py-1 rounded-md text-sm transition-colors ${
                mode === 'annotated'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setMode('annotated')}
            >
              Annotated
            </button>
            <button
              className={`px-3 py-1 rounded-md text-sm transition-colors ${
                mode === 'original'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setMode('original')}
            >
              Original
            </button>
          </div>
        )}
      </div>

      <div className="bg-black rounded-lg overflow-hidden">
        <video
          key={videoUrl}
          controls
          className="w-full max-h-[500px] mx-auto"
          preload="metadata"
        >
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>

      {mode === 'annotated' && (
        <div className="mt-3 flex gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm bg-blue-500" />
            Male
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-sm bg-pink-500" />
            Female
          </span>
          <span className="text-gray-400">
            Labels: #ID Gender AgeGroup
          </span>
        </div>
      )}
    </div>
  )
}

export default VideoPlayer
