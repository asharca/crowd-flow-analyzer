import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { getAnalytics, getVideo } from '../api/client'
import type { AnalyticsResponse } from '../types'
import AgeDistributionChart from '../components/AgeDistributionChart'
import FootTrafficChart from '../components/FootTrafficChart'
import GenderDistributionChart from '../components/GenderDistributionChart'
import VideoPlayer from '../components/VideoPlayer'

interface StatCardProps {
  label: string
  value: string | number
  color?: string
}

function StatCard({ label, value, color = 'text-gray-900' }: StatCardProps) {
  return (
    <div className="bg-white rounded-lg shadow p-4 text-center">
      <p className="text-sm text-gray-500">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
    </div>
  )
}

function SummaryCards({ analytics }: { analytics: AnalyticsResponse }) {
  const { gender_distribution: gd } = analytics
  const male = gd.male ?? 0
  const female = gd.female ?? 0
  const total = analytics.total_unique

  const peakTraffic = analytics.foot_traffic.reduce(
    (max, p) => (p.count > max ? p.count : max),
    0
  )

  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
      <StatCard label="Total Detected" value={total} />
      <StatCard label="Analyzed" value={analytics.total_analyzed} />
      <StatCard label="Male" value={male} color="text-blue-600" />
      <StatCard label="Female" value={female} color="text-pink-600" />
      <StatCard label="Peak Traffic" value={peakTraffic} color="text-orange-600" />
    </div>
  )
}

function DashboardPage() {
  const { id } = useParams<{ id: string }>()

  const videoQuery = useQuery({
    queryKey: ['video', id],
    queryFn: () => getVideo(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'processing' ? 3000 : false
    },
  })

  const analyticsQuery = useQuery({
    queryKey: ['analytics', id],
    queryFn: () => getAnalytics(id!),
    enabled: !!id && videoQuery.data?.status === 'completed',
  })

  if (videoQuery.isLoading) {
    return <p className="text-gray-500">Loading...</p>
  }

  const video = videoQuery.data
  if (!video) {
    return <p className="text-red-500">Video not found</p>
  }

  if (video.status !== 'completed') {
    return (
      <div className="text-center py-16">
        <p className="text-gray-500">
          Video is {video.status}
          {video.status === 'processing' && (
            <span className="ml-2 inline-block w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          )}
        </p>
        {video.error_message && (
          <p className="text-red-500 mt-2">{video.error_message}</p>
        )}
      </div>
    )
  }

  const analytics = analyticsQuery.data
  if (analyticsQuery.isLoading || !analytics) {
    return <p className="text-gray-500">Loading analytics...</p>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">{video.original_name}</h1>
        <div className="flex gap-4 mt-2 text-sm text-gray-500">
          {video.duration_sec && (
            <span>Duration: {video.duration_sec.toFixed(1)}s</span>
          )}
          {analytics.processing_time_sec && (
            <span>
              Processed in {analytics.processing_time_sec.toFixed(1)}s
            </span>
          )}
        </div>
      </div>

      {/* Summary Cards */}
      <SummaryCards analytics={analytics} />

      {/* Video Player */}
      <div className="bg-white rounded-lg shadow p-6">
        <VideoPlayer
          videoId={video.id}
          hasAnnotatedVideo={video.has_annotated_video}
        />
      </div>

      {/* Charts */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">
          Foot Traffic Over Time (by Gender)
        </h2>
        <FootTrafficChart data={analytics.foot_traffic} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">
            Age Distribution (by Gender)
          </h2>
          <AgeDistributionChart data={analytics.age_distribution} />
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Gender Distribution</h2>
          <GenderDistributionChart data={analytics.gender_distribution} />
        </div>
      </div>
    </div>
  )
}

export default DashboardPage
