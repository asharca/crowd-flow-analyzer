import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { deleteVideo, listVideos } from '../api/client'
import type { Video } from '../types'

const STATUS_COLORS: Record<string, string> = {
  queued: 'bg-yellow-100 text-yellow-800',
  processing: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString()
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_COLORS[status] ?? 'bg-gray-100 text-gray-800'}`}
    >
      {status}
    </span>
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
      const hasActive = videos.some(
        (v: Video) => v.status === 'queued' || v.status === 'processing'
      )
      return hasActive ? 3000 : false
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
      <div className="text-center py-16">
        <p className="text-gray-500 mb-4">No videos uploaded yet</p>
        <Link to="/" className="text-blue-600 hover:underline">
          Upload your first video
        </Link>
      </div>
    )
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Videos</h1>
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-gray-600">
            <tr>
              <th className="px-4 py-3">Name</th>
              <th className="px-4 py-3">Size</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Uploaded</th>
              <th className="px-4 py-3">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {videos.map((video) => (
              <tr key={video.id} className="hover:bg-gray-50">
                <td className="px-4 py-3">
                  {video.status === 'completed' ? (
                    <Link
                      to={`/videos/${video.id}`}
                      className="text-blue-600 hover:underline"
                    >
                      {video.original_name}
                    </Link>
                  ) : (
                    video.original_name
                  )}
                </td>
                <td className="px-4 py-3 text-gray-500">
                  {formatSize(video.file_size)}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={video.status} />
                  {video.status === 'processing' && (
                    <span className="ml-2 inline-block w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  )}
                </td>
                <td className="px-4 py-3 text-gray-500">
                  {formatTime(video.created_at)}
                </td>
                <td className="px-4 py-3">
                  <button
                    className="text-red-500 hover:text-red-700 text-xs"
                    onClick={() => {
                      if (confirm('Delete this video?')) {
                        deleteMutation.mutate(video.id)
                      }
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default VideoListPage
