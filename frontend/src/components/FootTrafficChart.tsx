import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { FootTrafficPoint } from '../types'

interface FootTrafficChartProps {
  data: FootTrafficPoint[]
}

function FootTrafficChart({ data }: FootTrafficChartProps) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="timestamp_sec"
          label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
        />
        <YAxis
          label={{
            value: 'People Count',
            angle: -90,
            position: 'insideLeft',
          }}
        />
        <Tooltip labelFormatter={(label) => `${label}s`} />
        <Legend />
        <Area
          type="monotone"
          dataKey="count"
          name="Total"
          stroke="#6b7280"
          fill="#e5e7eb"
          strokeWidth={2}
          fillOpacity={0.2}
          dot={false}
        />
        <Area
          type="monotone"
          dataKey="male"
          name="Male"
          stroke="#3b82f6"
          fill="#93c5fd"
          strokeWidth={2}
          fillOpacity={0.3}
          dot={false}
        />
        <Area
          type="monotone"
          dataKey="female"
          name="Female"
          stroke="#ec4899"
          fill="#f9a8d4"
          strokeWidth={2}
          fillOpacity={0.3}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default FootTrafficChart
