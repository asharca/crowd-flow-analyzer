import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { AgeGroupDetail } from '../types'

interface AgeDistributionChartProps {
  data: Record<string, AgeGroupDetail>
}

const AGE_ORDER = ['0-18', '19-30', '31-45', '46-60', '60+']

function AgeDistributionChart({ data }: AgeDistributionChartProps) {
  const chartData = AGE_ORDER.map((group) => {
    const detail = data[group]
    return {
      group,
      male: detail?.male ?? 0,
      female: detail?.female ?? 0,
      total: detail?.total ?? 0,
    }
  })

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="group" />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Legend />
        <Bar
          dataKey="male"
          name="Male"
          stackId="gender"
          fill="#3b82f6"
          radius={[0, 0, 0, 0]}
        />
        <Bar
          dataKey="female"
          name="Female"
          stackId="gender"
          fill="#ec4899"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default AgeDistributionChart
