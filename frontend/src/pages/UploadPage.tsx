import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadVideo } from '../api/client'

function UploadPage() {
  const navigate = useNavigate()
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
        await uploadVideo(file)
        navigate('/videos')
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message)
        } else {
          setError('Upload failed')
        }
      } finally {
        setUploading(false)
      }
    },
    [navigate]
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
    <div>
      <h1 className="text-2xl font-bold mb-6">Upload Video</h1>
      <div
        className={`border-2 border-dashed rounded-xl p-16 text-center transition-colors ${
          dragging
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 bg-white hover:border-gray-400'
        }`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
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
            <p className="text-gray-500 mb-4">
              Drag and drop a video file here, or click to browse
            </p>
            <label className="inline-block px-6 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors">
              Choose File
              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv"
                className="hidden"
                onChange={handleInputChange}
              />
            </label>
            <p className="text-xs text-gray-400 mt-3">
              Supported: MP4, AVI, MOV, MKV (max 500MB)
            </p>
          </>
        )}
      </div>
      {error && (
        <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">
          {error}
        </div>
      )}
    </div>
  )
}

export default UploadPage
