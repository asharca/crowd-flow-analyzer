export interface Video {
  id: string
  original_name: string
  file_size: number
  duration_sec: number | null
  status: 'queued' | 'processing' | 'completed' | 'failed'
  error_message: string | null
  has_annotated_video: boolean
  created_at: string
  completed_at: string | null
}

export interface VideoListResponse {
  videos: Video[]
  total: number
}

export interface FootTrafficPoint {
  timestamp_sec: number
  count: number
  male: number
  female: number
  unknown: number
}

export interface AgeGroupDetail {
  male: number
  female: number
  total: number
}

export interface PersonResult {
  track_id: number
  age: number | null
  age_group: string | null
  gender: string
  confidence: number
}

export interface PipelineConfig {
  device: string
  gpu_name?: string
  gpu_vram_gb?: number
  yolo_model: string
  demographics_model: string
  yolo_batch_size: number
  mivolo_batch_size?: number
  frame_skip?: number
}

export interface AnalyticsResponse {
  video_id: string
  total_unique: number
  total_analyzed: number
  foot_traffic: FootTrafficPoint[]
  age_distribution: Record<string, AgeGroupDetail>
  gender_distribution: Record<string, number>
  persons: PersonResult[]
  pipeline_config: PipelineConfig
  processing_time_sec: number | null
}

export interface UploadResponse {
  id: string
  status: string
  message: string
}

export interface SystemInfo {
  device: string
  gpu_name?: string
  gpu_vram_gb?: number
  yolo_model: string
  demographics_model: string
  yolo_batch_size: number
  mivolo_batch_size?: number
  frame_skip?: number
}

export interface ProgressResponse {
  stage: string
  stage_percent: number
  overall_percent: number
  detail: string
  system_info: SystemInfo | Record<string, never>
}
