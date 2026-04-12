import { useCallback, useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { getModels, uploadVideo } from '../api/client'
import type { ModelOption, ServerDefaults } from '../api/client'

const SIZE_ORDER: Record<string, number> = { nano: 0, small: 1, medium: 2, large: 3, xlarge: 4 }

/* ── Model Selector ──────────────────────────────────────────── */

function ModelSelector({
  models,
  defaultId,
  selected,
  onSelect,
}: {
  models: ModelOption[]
  defaultId: string
  selected: string
  onSelect: (id: string) => void
}) {
  const families = [...new Set(models.map((m) => m.family))]

  return (
    <div className="space-y-4">
      {families.map((family) => {
        const familyModels = models
          .filter((m) => m.family === family)
          .sort((a, b) => (SIZE_ORDER[a.size] ?? 0) - (SIZE_ORDER[b.size] ?? 0))
        return (
          <div key={family}>
            <p className="text-xs font-medium text-gray-500 mb-2">{family}</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
              {familyModels.map((m) => {
                const isSelected = selected === m.id
                const isDefault = defaultId === m.id
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => onSelect(m.id)}
                    className={`relative text-left rounded-lg border p-3 transition-all text-sm ${
                      isSelected
                        ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500'
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    {isDefault && (
                      <span className="absolute -top-2 right-2 text-[10px] px-1.5 py-0.5 rounded-full bg-green-100 text-green-700 font-medium">
                        default
                      </span>
                    )}
                    <p className={`font-medium ${isSelected ? 'text-blue-700' : 'text-gray-800'}`}>
                      {m.size.charAt(0).toUpperCase() + m.size.slice(1)}
                    </p>
                    <div className="mt-1 space-y-0.5 text-xs text-gray-500">
                      <p>mAP: <span className="font-medium text-gray-700">{m.map50_95}</span></p>
                      <p>{m.params_m}M params</p>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ── Slider Parameter ────────────────────────────────────────── */

interface SliderParamProps {
  label: string
  description: string
  value: number
  defaultValue: number
  min: number
  max: number
  step: number
  unit?: string
  onChange: (v: number) => void
}

function SliderParam({ label, description, value, defaultValue, min, max, step, unit, onChange }: SliderParamProps) {
  const isCustom = value !== defaultValue
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-sm font-medium text-gray-700">{label}</span>
          <span className="text-xs text-gray-400 ml-2">{description}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-mono font-medium ${isCustom ? 'text-blue-600' : 'text-gray-600'}`}>
            {value}{unit ?? ''}
          </span>
          {isCustom && (
            <button
              type="button"
              onClick={() => onChange(defaultValue)}
              className="text-[10px] text-gray-400 hover:text-gray-600 underline"
            >
              reset
            </button>
          )}
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-blue-600"
      />
      <div className="flex justify-between text-[10px] text-gray-400">
        <span>{min}{unit ?? ''}</span>
        <span className="text-gray-500">default: {defaultValue}{unit ?? ''}</span>
        <span>{max}{unit ?? ''}</span>
      </div>
    </div>
  )
}

/* ── Main Upload Page ────────────────────────────────────────── */

function UploadPage() {
  const navigate = useNavigate()
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Model + params state
  const [selectedModel, setSelectedModel] = useState('')
  const [frameSkip, setFrameSkip] = useState(0)
  const [yoloBatch, setYoloBatch] = useState(0)
  const [mivoloBatch, setMivoloBatch] = useState(0)
  const [maxCrops, setMaxCrops] = useState(0)

  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    staleTime: 60_000,
  })

  // Initialize with server defaults once loaded
  useEffect(() => {
    if (modelsData && !selectedModel) {
      setSelectedModel(modelsData.default)
      const d = modelsData.defaults
      setFrameSkip(d.frame_skip)
      setYoloBatch(d.yolo_batch_size)
      setMivoloBatch(d.mivolo_batch_size)
      setMaxCrops(d.max_crops)
    }
  }, [modelsData, selectedModel])

  const defaults: ServerDefaults | undefined = modelsData?.defaults
  const isGpu = defaults?.device?.startsWith('cuda')

  const handleFile = useCallback(
    async (file: File) => {
      const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
      if (!['mp4', 'avi', 'mov', 'mkv'].includes(ext)) {
        setError(`Unsupported format: .${ext}`)
        return
      }

      setError(null)
      setUploading(true)
      try {
        // Only send non-default values (0 = let server decide)
        const params = {
          model: selectedModel || undefined,
          frame_skip: defaults && frameSkip !== defaults.frame_skip ? frameSkip : undefined,
          yolo_batch_size: defaults && yoloBatch !== defaults.yolo_batch_size ? yoloBatch : undefined,
          mivolo_batch_size: defaults && mivoloBatch !== defaults.mivolo_batch_size ? mivoloBatch : undefined,
          max_crops: defaults && maxCrops !== defaults.max_crops ? maxCrops : undefined,
        }
        const res = await uploadVideo(file, params)
        navigate(`/videos/${res.id}`)
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Upload failed')
      } finally {
        setUploading(false)
      }
    },
    [navigate, selectedModel, frameSkip, yoloBatch, mivoloBatch, maxCrops, defaults]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Upload Video</h1>
        {defaults && (
          <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
            <span className={`w-1.5 h-1.5 rounded-full ${isGpu ? 'bg-green-500' : 'bg-orange-400'}`} />
            Server: {isGpu ? 'GPU' : 'CPU'}
          </span>
        )}
      </div>

      {/* Detection Model */}
      {modelsData && (
        <div className="bg-white rounded-xl border border-gray-100 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Detection Model</h3>
            <span className="text-xs text-gray-400">Higher mAP = more accurate</span>
          </div>
          <ModelSelector
            models={modelsData.models}
            defaultId={modelsData.default}
            selected={selectedModel}
            onSelect={setSelectedModel}
          />
        </div>
      )}

      {/* Advanced Parameters (collapsible) */}
      {defaults && (
        <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full px-5 py-4 flex items-center justify-between text-sm font-semibold text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
              </svg>
              Pipeline Parameters
            </span>
            <svg className={`w-4 h-4 text-gray-400 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
          </button>

          {showAdvanced && (
            <div className="px-5 pb-5 space-y-6 border-t border-gray-50 pt-4">
              <SliderParam
                label="Frame Skip"
                description="Process every Nth frame"
                value={frameSkip}
                defaultValue={defaults.frame_skip}
                min={1}
                max={10}
                step={1}
                onChange={setFrameSkip}
              />
              <SliderParam
                label="YOLO Batch Size"
                description="Frames per GPU batch"
                value={yoloBatch}
                defaultValue={defaults.yolo_batch_size}
                min={4}
                max={256}
                step={4}
                onChange={setYoloBatch}
              />
              <SliderParam
                label="MiVOLO Batch Size"
                description="Body crops per GPU batch"
                value={mivoloBatch}
                defaultValue={defaults.mivolo_batch_size}
                min={8}
                max={256}
                step={8}
                onChange={setMivoloBatch}
              />
              <SliderParam
                label="Max Crops per Person"
                description="More = better accuracy, slower"
                value={maxCrops}
                defaultValue={defaults.max_crops}
                min={1}
                max={10}
                step={1}
                onChange={setMaxCrops}
              />
            </div>
          )}
        </div>
      )}

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-14 text-center transition-colors ${
          dragging
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 bg-white hover:border-gray-400'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {uploading ? (
          <div className="text-gray-500">
            <div className="animate-spin inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mb-3" />
            <p>Uploading...</p>
          </div>
        ) : (
          <>
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-gray-100 mb-4">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <p className="text-gray-600 mb-2">Drag and drop a video file, or click to browse</p>
            <label className="inline-block px-6 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors text-sm font-medium">
              Choose File
              <input type="file" accept=".mp4,.avi,.mov,.mkv" className="hidden" onChange={handleInputChange} />
            </label>
            <p className="text-xs text-gray-400 mt-3">MP4, AVI, MOV, MKV (max 500MB)</p>
          </>
        )}
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{error}</div>
      )}
    </div>
  )
}

export default UploadPage
