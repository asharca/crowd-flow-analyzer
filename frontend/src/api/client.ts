import axios from 'axios'
import type {
  AnalyticsResponse,
  ProgressResponse,
  UploadResponse,
  Video,
  VideoListResponse,
} from '../types'

const api = axios.create({ baseURL: '/api' })

export interface ModelOption {
  id: string
  name: string
  family: string
  size: string
  params_m: number
  map50_95: number
  recommended: boolean
}

export interface ServerDefaults {
  frame_skip: number
  yolo_batch_size: number
  mivolo_batch_size: number
  max_crops: number
  device: string
}

export interface ModelsResponse {
  models: ModelOption[]
  default: string
  defaults: ServerDefaults
}

export async function getModels(): Promise<ModelsResponse> {
  const { data } = await api.get<ModelsResponse>('/videos/models')
  return data
}

export interface PipelineParams {
  model?: string
  frame_skip?: number
  yolo_batch_size?: number
  mivolo_batch_size?: number
  max_crops?: number
}

export async function uploadVideo(file: File, params?: PipelineParams): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  if (params?.model) formData.append('model', params.model)
  if (params?.frame_skip) formData.append('frame_skip', String(params.frame_skip))
  if (params?.yolo_batch_size) formData.append('yolo_batch_size', String(params.yolo_batch_size))
  if (params?.mivolo_batch_size) formData.append('mivolo_batch_size', String(params.mivolo_batch_size))
  if (params?.max_crops) formData.append('max_crops', String(params.max_crops))
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

export async function getProgress(
  videoId: string
): Promise<ProgressResponse> {
  const { data } = await api.get<ProgressResponse>(
    `/videos/${videoId}/progress`
  )
  return data
}
