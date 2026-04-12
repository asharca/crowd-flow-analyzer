import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { deleteVideo, listVideos } from '../api/client'
import type { Video } from '../types'

const STATUS_STYLES: Record<string, { bg: string; dot: string }> = {
  queued: { bg: 'bg-yellow-50 text-yellow-700 ring-yellow-600/20', dot: 'bg-yellow-500' },
  processing: { bg: 'bg-blue-50 text-blue-700 ring-blue-600/20', dot: 'bg-blue-500' },
  completed: { bg: 'bg-green-50 text-green-700 ring-green-600/20', dot: 'bg-green-500' },
  failed: { bg: 'bg-red-50 text-red-700 ring-red-600/20', dot: 'bg-red-500' },
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMin = Math.floor(diffMs / 60000)
  if (diffMin < 1) return 'Just now'
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  return d.toLocaleDateString()
}

function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLES[status] ?? STATUS_STYLES.queued
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ring-1 ring-inset ${s.bg}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${status === 'processing' ? 'animate-pulse' : ''}`} />
      {status}
    </span>
  )
}

function VideoCard({ video, onDelete }: { video: Video; onDelete: (id: string) => void }) {
  return (
    <Link
      to={`/videos/${video.id}`}
      className="block bg-white rounded-xl border border-gray-100 hover:border-gray-200 hover:shadow-md transition-all p-5"
    >
      <div className="flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <h3 className="font-medium text-gray-900 truncate">{video.original_name}</h3>
          <div className="flex items-center gap-3 mt-1.5 text-xs text-gray-500">
            <span>{formatSize(video.file_size)}</span>
            {video.duration_sec && <span>{video.duration_sec.toFixed(1)}s</span>}
            <span>{formatTime(video.created_at)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 ml-4 flex-shrink-0">
          <StatusBadge status={video.status} />
          <button
            className="p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-500 transition-colors"
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              if (confirm('Delete this video?')) onDelete(video.id)
            }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
            </svg>
          </button>
        </div>
      </div>
    </Link>
  )
}

function VideoListPage() {
  const queryClient = useQueryClient()
  const { data, isLoading } = useQuery({
    queryKey: ['videos'],
    queryFn: () => listVideos(),
    refetchInterval: (query) => {
      const videos = query.state.data?.videos
      if (!videos) return false
      return videos.some((v: Video) => v.status === 'queued' || v.status === 'processing') ? 3000 : false
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteVideo,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['videos'] }),
  })

  if (isLoading) {
    return <p className="text-gray-500">Loading...</p>
  }

  const videos = data?.videos ?? []

  if (videos.length === 0) {
    return (
      <div className="text-center py-20">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mb-4">
          <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
          </svg>
        </div>
        <p className="text-gray-600 font-medium mb-1">No videos yet</p>
        <p className="text-sm text-gray-400 mb-4">Upload a video to get started</p>
        <Link to="/" className="text-sm text-blue-600 hover:text-blue-700 font-medium">
          Upload Video
        </Link>
      </div>
    )
  }

  const completed = videos.filter((v) => v.status === 'completed').length
  const processing = videos.filter((v) => v.status === 'processing' || v.status === 'queued').length

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Videos</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {videos.length} video{videos.length !== 1 ? 's' : ''}
            {completed > 0 && <span> &middot; {completed} completed</span>}
            {processing > 0 && <span> &middot; {processing} in progress</span>}
          </p>
        </div>
        <Link
          to="/"
          className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
        >
          Upload
        </Link>
      </div>
      <div className="space-y-3">
        {videos.map((video) => (
          <VideoCard key={video.id} video={video} onDelete={(vid) => deleteMutation.mutate(vid)} />
        ))}
      </div>
    </div>
  )
}

export default VideoListPage
