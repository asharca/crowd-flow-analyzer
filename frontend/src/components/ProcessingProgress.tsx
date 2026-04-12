import { useQuery } from '@tanstack/react-query'
import { getProgress } from '../api/client'
import type { ProgressResponse } from '../types'

interface ProcessingProgressProps {
  videoId: string
  videoName: string
}

const STAGES = [
  { key: 'detection', label: 'Person Detection', icon: 'YOLO' },
  { key: 'tracking', label: 'Multi-Object Tracking', icon: 'ByteTrack' },
  { key: 'demographics', label: 'Age & Gender Analysis', icon: 'MiVOLO' },
  { key: 'annotation', label: 'Video Annotation', icon: 'OpenCV' },
  { key: 'aggregation', label: 'Statistics', icon: 'Aggregate' },
]

function StageTimeline({ currentStage }: { currentStage: string }) {
  const currentIdx = STAGES.findIndex((s) => s.key === currentStage)

  return (
    <div className="space-y-3">
      {STAGES.map((stage, idx) => {
        const isActive = stage.key === currentStage
        const isDone = idx < currentIdx || currentStage === 'completed'
        const isPending = idx > currentIdx && currentStage !== 'completed'

        return (
          <div key={stage.key} className="flex items-center gap-3">
            {/* Status indicator */}
            <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center">
              {isDone ? (
                <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                  <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              ) : isActive ? (
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                  <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                </div>
              ) : (
                <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-gray-300" />
                </div>
              )}
            </div>

            {/* Label */}
            <div className="flex-1 min-w-0">
              <p className={`text-sm font-medium ${
                isActive ? 'text-blue-700' : isDone ? 'text-green-700' : 'text-gray-400'
              }`}>
                {stage.label}
              </p>
            </div>

            {/* Engine badge */}
            <span className={`text-xs px-2 py-0.5 rounded-full ${
              isPending
                ? 'bg-gray-100 text-gray-400'
                : 'bg-gray-100 text-gray-600'
            }`}>
              {stage.icon}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function SystemInfoPanel({ progress }: { progress: ProgressResponse }) {
  const info = progress.system_info
  if (!info || !('device' in info)) return null

  const isGpu = info.device?.startsWith('cuda')

  return (
    <div className="bg-gray-50 rounded-lg p-4 space-y-2">
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">System Info</h4>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
        <span className="text-gray-500">Device</span>
        <span className="font-medium">
          {isGpu ? (
            <span className="text-green-700">
              GPU {info.gpu_name && `(${info.gpu_name})`}
            </span>
          ) : (
            <span className="text-orange-600">CPU</span>
          )}
        </span>

        {isGpu && info.gpu_vram_gb && (
          <>
            <span className="text-gray-500">VRAM</span>
            <span className="font-medium">{info.gpu_vram_gb} GB</span>
          </>
        )}

        <span className="text-gray-500">Detection</span>
        <span className="font-medium">{info.yolo_model} (batch {info.yolo_batch_size})</span>

        <span className="text-gray-500">Demographics</span>
        <span className="font-medium">{info.demographics_model} (batch {info.mivolo_batch_size ?? 32})</span>

        {info.frame_skip && (
          <>
            <span className="text-gray-500">Frame Skip</span>
            <span className="font-medium">
              {info.frame_skip === 1 ? 'Every frame' : `Every ${info.frame_skip} frames`}
            </span>
          </>
        )}
      </div>
    </div>
  )
}

function ProcessingProgress({ videoId, videoName }: ProcessingProgressProps) {
  const { data: progress } = useQuery({
    queryKey: ['progress', videoId],
    queryFn: () => getProgress(videoId),
    refetchInterval: 1500,
  })

  const overall = progress?.overall_percent ?? 0
  const stage = progress?.stage ?? 'queued'
  const detail = progress?.detail ?? 'Waiting in queue...'

  return (
    <div className="max-w-lg mx-auto py-12 space-y-8">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-blue-50 mb-2">
          <svg className="w-8 h-8 text-blue-600 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
          </svg>
        </div>
        <h2 className="text-xl font-bold text-gray-900">Analyzing Video</h2>
        <p className="text-sm text-gray-500 truncate max-w-xs mx-auto">{videoName}</p>
      </div>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600 font-medium">{detail}</span>
          <span className="text-blue-600 font-bold">{overall}%</span>
        </div>
        <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-700 ease-out"
            style={{ width: `${overall}%` }}
          />
        </div>
      </div>

      {/* Stage timeline */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Pipeline Progress</h3>
        <StageTimeline currentStage={stage} />
      </div>

      {/* System info */}
      {progress && <SystemInfoPanel progress={progress} />}
    </div>
  )
}

export default ProcessingProgress
