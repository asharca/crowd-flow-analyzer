import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { getAnalytics, getVideo } from '../api/client'
import type { AnalyticsResponse, PipelineConfig, PersonResult } from '../types'
import AgeDistributionChart from '../components/AgeDistributionChart'
import FootTrafficChart from '../components/FootTrafficChart'
import GenderDistributionChart from '../components/GenderDistributionChart'
import ProcessingProgress from '../components/ProcessingProgress'
import VideoPlayer from '../components/VideoPlayer'

/* ── Stat Card ─────────────────────────────────────────────────── */

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  icon: React.ReactNode
  accent: string
}

function StatCard({ label, value, sub, icon, accent }: StatCardProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 p-5 flex items-start gap-4">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${accent}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</p>
        <p className="text-2xl font-bold text-gray-900 mt-0.5">{value}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

/* ── Pipeline Config Panel ──────────────────────────────────────── */

function PipelinePanel({ config, processingTime }: { config: PipelineConfig; processingTime: number | null }) {
  const isGpu = config.device?.startsWith('cuda')
  const items: { label: string; value: string }[] = [
    {
      label: 'Compute',
      value: isGpu
        ? `GPU - ${config.gpu_name ?? 'CUDA'}${config.gpu_vram_gb ? ` (${config.gpu_vram_gb} GB)` : ''}`
        : 'CPU',
    },
    { label: 'Detection', value: `${config.yolo_model} (batch ${config.yolo_batch_size})` },
    { label: 'Demographics', value: `${config.demographics_model}${config.mivolo_batch_size ? ` (batch ${config.mivolo_batch_size})` : ''}` },
  ]
  if (config.frame_skip) {
    items.push({ label: 'Sampling', value: config.frame_skip === 1 ? 'Every frame' : `Every ${config.frame_skip} frames` })
  }
  if (processingTime) {
    items.push({ label: 'Processing', value: `${processingTime.toFixed(1)}s` })
  }

  return (
    <div className="bg-white rounded-xl border border-gray-100 p-5">
      <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 010-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28z" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        Pipeline Configuration
      </h3>
      <dl className="space-y-2">
        {items.map((item) => (
          <div key={item.label} className="flex items-center justify-between text-sm">
            <dt className="text-gray-500">{item.label}</dt>
            <dd className="font-medium text-gray-800">{item.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

/* ── Persons Table ──────────────────────────────────────────────── */

const GENDER_DOT: Record<string, string> = {
  male: 'bg-blue-500',
  female: 'bg-pink-500',
}

function PersonsTable({ persons }: { persons: PersonResult[] }) {
  const [sortKey, setSortKey] = useState<'track_id' | 'age' | 'confidence'>('track_id')
  const [sortAsc, setSortAsc] = useState(true)
  const [filterGender, setFilterGender] = useState<string>('all')

  const filtered = filterGender === 'all' ? persons : persons.filter((p) => p.gender === filterGender)

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey] ?? 0
    const bv = b[sortKey] ?? 0
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
  })

  function toggleSort(key: typeof sortKey) {
    if (sortKey === key) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(true) }
  }

  const SortIcon = ({ col }: { col: typeof sortKey }) => {
    if (sortKey !== col) return <span className="text-gray-300 ml-1">&#8597;</span>
    return <span className="text-blue-600 ml-1">{sortAsc ? '\u2191' : '\u2193'}</span>
  }

  return (
    <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
      <div className="px-5 py-4 flex items-center justify-between border-b border-gray-50">
        <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
          </svg>
          Individual Results
          <span className="text-gray-400 font-normal">({filtered.length})</span>
        </h3>
        <div className="flex gap-1 text-xs">
          {['all', 'male', 'female', 'unknown'].map((g) => (
            <button
              key={g}
              onClick={() => setFilterGender(g)}
              className={`px-2.5 py-1 rounded-full transition-colors ${
                filterGender === g
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {g === 'all' ? 'All' : g.charAt(0).toUpperCase() + g.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="max-h-80 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left text-gray-500 sticky top-0">
            <tr>
              <th className="px-5 py-2.5 font-medium cursor-pointer select-none" onClick={() => toggleSort('track_id')}>
                ID <SortIcon col="track_id" />
              </th>
              <th className="px-5 py-2.5 font-medium">Gender</th>
              <th className="px-5 py-2.5 font-medium cursor-pointer select-none" onClick={() => toggleSort('age')}>
                Age <SortIcon col="age" />
              </th>
              <th className="px-5 py-2.5 font-medium">Age Group</th>
              <th className="px-5 py-2.5 font-medium cursor-pointer select-none" onClick={() => toggleSort('confidence')}>
                Confidence <SortIcon col="confidence" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {sorted.map((p) => (
              <tr key={p.track_id} className="hover:bg-gray-50/50">
                <td className="px-5 py-2 font-mono text-gray-600">#{p.track_id}</td>
                <td className="px-5 py-2">
                  <span className="inline-flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${GENDER_DOT[p.gender] ?? 'bg-gray-400'}`} />
                    {p.gender === 'unknown' ? '-' : p.gender}
                  </span>
                </td>
                <td className="px-5 py-2">{p.age ?? '-'}</td>
                <td className="px-5 py-2">
                  {p.age_group ? (
                    <span className="px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700">{p.age_group}</span>
                  ) : '-'}
                </td>
                <td className="px-5 py-2">
                  {p.confidence > 0 ? (
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${p.confidence >= 0.8 ? 'bg-green-500' : p.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-400'}`}
                          style={{ width: `${p.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-gray-500 text-xs">{(p.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ) : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ── Summary Stats ──────────────────────────────────────────────── */

function SummaryCards({ analytics }: { analytics: AnalyticsResponse }) {
  const { gender_distribution: gd } = analytics
  const male = gd.male ?? 0
  const female = gd.female ?? 0
  const peakTraffic = analytics.foot_traffic.reduce(
    (max, p) => (p.count > max ? p.count : max), 0
  )
  const analyzedWithConf = analytics.persons.filter((p) => p.confidence > 0)
  const avgConfidence = analyzedWithConf.length > 0
    ? analyzedWithConf.reduce((sum, p) => sum + p.confidence, 0) / analyzedWithConf.length
    : 0

  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
      <StatCard
        label="Total Detected"
        value={analytics.total_unique}
        icon={<svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" /></svg>}
        accent="bg-indigo-50"
      />
      <StatCard
        label="Analyzed"
        value={analytics.total_analyzed}
        sub={`${analytics.total_unique > 0 ? Math.round(analytics.total_analyzed / analytics.total_unique * 100) : 0}% coverage`}
        icon={<svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
        accent="bg-green-50"
      />
      <StatCard
        label="Male"
        value={male}
        icon={<svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>}
        accent="bg-blue-50"
      />
      <StatCard
        label="Female"
        value={female}
        icon={<svg className="w-5 h-5 text-pink-600" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>}
        accent="bg-pink-50"
      />
      <StatCard
        label="Peak Traffic"
        value={peakTraffic}
        sub={avgConfidence > 0 ? `Avg conf: ${(avgConfidence * 100).toFixed(0)}%` : undefined}
        icon={<svg className="w-5 h-5 text-orange-600" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941" /></svg>}
        accent="bg-orange-50"
      />
    </div>
  )
}

/* ── Tab Selector ───────────────────────────────────────────────── */

type TabKey = 'overview' | 'people' | 'config'

function TabBar({ active, onChange }: { active: TabKey; onChange: (t: TabKey) => void }) {
  const tabs: { key: TabKey; label: string }[] = [
    { key: 'overview', label: 'Overview' },
    { key: 'people', label: 'People' },
    { key: 'config', label: 'Pipeline' },
  ]

  return (
    <div className="flex gap-1 bg-gray-100 rounded-lg p-1 w-fit">
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            active === t.key
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

/* ── Main Dashboard ─────────────────────────────────────────────── */

function DashboardPage() {
  const { id } = useParams<{ id: string }>()
  const [tab, setTab] = useState<TabKey>('overview')

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

  /* Processing / Failed states */
  if (video.status !== 'completed') {
    if (video.status === 'failed') {
      return (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-50 mb-4">
            <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <p className="text-red-600 font-medium">Processing Failed</p>
          {video.error_message && (
            <p className="text-red-400 mt-2 text-sm max-w-md mx-auto">{video.error_message}</p>
          )}
        </div>
      )
    }
    return <ProcessingProgress videoId={video.id} videoName={video.original_name} />
  }

  const analytics = analyticsQuery.data
  if (analyticsQuery.isLoading || !analytics) {
    return <p className="text-gray-500">Loading analytics...</p>
  }

  return (
    <div className="space-y-6">
      {/* Breadcrumb + Header */}
      <div>
        <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
          <Link to="/videos" className="hover:text-gray-700">Videos</Link>
          <span>/</span>
          <span className="text-gray-800">{video.original_name}</span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{video.original_name}</h1>
            <div className="flex flex-wrap gap-3 mt-2 text-sm text-gray-500">
              {video.duration_sec && <span>Duration: {video.duration_sec.toFixed(1)}s</span>}
              {analytics.processing_time_sec && (
                <span>Processed in {analytics.processing_time_sec.toFixed(1)}s</span>
              )}
              <span className="inline-flex items-center gap-1">
                <span className={`w-1.5 h-1.5 rounded-full ${analytics.pipeline_config.device?.startsWith('cuda') ? 'bg-green-500' : 'bg-orange-400'}`} />
                {analytics.pipeline_config.device?.startsWith('cuda') ? 'GPU' : 'CPU'}
              </span>
            </div>
          </div>
          <TabBar active={tab} onChange={setTab} />
        </div>
      </div>

      {/* Summary Cards (always visible) */}
      <SummaryCards analytics={analytics} />

      {/* Tab Content */}
      {tab === 'overview' && (
        <>
          <div className="bg-white rounded-xl border border-gray-100 p-6">
            <VideoPlayer videoId={video.id} hasAnnotatedVideo={video.has_annotated_video} />
          </div>

          <div className="bg-white rounded-xl border border-gray-100 p-6">
            <h2 className="text-base font-semibold text-gray-800 mb-4">Foot Traffic Over Time</h2>
            <FootTrafficChart data={analytics.foot_traffic} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-xl border border-gray-100 p-6">
              <h2 className="text-base font-semibold text-gray-800 mb-4">Age Distribution</h2>
              <AgeDistributionChart data={analytics.age_distribution} />
            </div>
            <div className="bg-white rounded-xl border border-gray-100 p-6">
              <h2 className="text-base font-semibold text-gray-800 mb-4">Gender Distribution</h2>
              <GenderDistributionChart data={analytics.gender_distribution} />
            </div>
          </div>
        </>
      )}

      {tab === 'people' && (
        <PersonsTable persons={analytics.persons} />
      )}

      {tab === 'config' && (
        <div className="max-w-lg">
          <PipelinePanel config={analytics.pipeline_config} processingTime={analytics.processing_time_sec} />
        </div>
      )}
    </div>
  )
}

export default DashboardPage
