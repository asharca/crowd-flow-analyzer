import axios from 'axios'
import type {
  AnalyticsResponse,
  UploadResponse,
  Video,
  VideoListResponse,
} from '../types'

const api = axios.create({ baseURL: '/api' })

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post<UploadResponse>('/videos/upload', formData)
  return data
}

export async function listVideos(
  status?: string
): Promise<VideoListResponse> {
  const params = status ? { status } : {}
  const { data } = await api.get<VideoListResponse>('/videos', { params })
  return data
}

export async function getVideo(id: string): Promise<Video> {
  const { data } = await api.get<Video>(`/videos/${id}`)
  return data
}

export async function deleteVideo(id: string): Promise<void> {
  await api.delete(`/videos/${id}`)
}

export async function getAnalytics(
  videoId: string
): Promise<AnalyticsResponse> {
  const { data } = await api.get<AnalyticsResponse>(
    `/videos/${videoId}/analytics`
  )
  return data
}
